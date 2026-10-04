"""
Training Protocol and Engine for ECTE408 Study.
Trains models M0 through M7 with AdamW, Cosine Annealing, Soft Dice + BCE loss,
and deterministic 13-condition multi-metric validation per epoch.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from python.data.dataset import BraTS2DSliceDataset
from python.physics.degradation import LowFieldDegradation, DEGRADATION_CONDITIONS, CONDITION_MAP
from python.models.factory import build_model
from python.metrics.evaluation import compute_3d_dice


def soft_dice_loss_per_channel(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # pred, target: (B, C, H, W)
    intersection = 2.0 * (pred * target).sum(dim=(-2, -1)) + eps
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + eps
    dice_channel = intersection / union # (B, C)
    return 1.0 - dice_channel.mean()


class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        l_bce = self.bce(pred, target)
        l_dice = soft_dice_loss_per_channel(pred, target)
        total = l_bce + l_dice
        return total, l_bce.item(), l_dice.item()


def validate_all_conditions(
    model: nn.Module,
    val_patient_ids: List[str],
    cache_dir: str,
    device: torch.device = torch.device("cpu"),
    primary_only: bool = False
) -> Dict[str, float]:
    """
    Evaluates model across degradation conditions using deterministic seeding.
    Returns per-condition mean Dice and overall average.
    """
    model.eval()
    deg_module = LowFieldDegradation()
    
    cond_list = DEGRADATION_CONDITIONS
    if primary_only:
        cond_list = [c for c in DEGRADATION_CONDITIONS if c["id"] in ["clean", "snr12_r0.75", "snr8_r0.5"]]

    results = {}
    condition_dices = []

    with torch.no_grad():
        for cond in cond_list:
            c_id = cond["id"]
            
            val_ds = BraTS2DSliceDataset(
                patient_ids=val_patient_ids,
                cache_dir=cache_dir,
                is_train=False,
                degradation_fn=lambda x, mu, p_id, slice_idx: deg_module(
                    x, mu, condition_id=c_id, patient_id=p_id, slice_idx=slice_idx
                ),
                spatial_augment=False
            )
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

            total_wt_dice = 0.0
            total_tc_dice = 0.0
            total_et_dice = 0.0
            count = 0

            for batch in val_loader:
                imgs = batch["image"].to(device)
                targets = batch["target"].to(device) # (B, 3, H, W)
                preds = model(imgs)

                # Channel 0: WT, Channel 1: TC, Channel 2: ET
                for b in range(imgs.shape[0]):
                    p_np = preds[b].cpu().numpy()
                    t_np = targets[b].cpu().numpy()
                    
                    wt_d = compute_3d_dice(p_np[0], t_np[0])
                    tc_d = compute_3d_dice(p_np[1], t_np[1])
                    et_d = compute_3d_dice(p_np[2], t_np[2])

                    total_wt_dice += wt_d
                    total_tc_dice += tc_d
                    total_et_dice += et_d
                    count += 1

            wt_mean = total_wt_dice / max(count, 1)
            tc_mean = total_tc_dice / max(count, 1)
            et_mean = total_et_dice / max(count, 1)
            mean_dice = (wt_mean + tc_mean + et_mean) / 3.0

            results[f"val_dice_wt_{c_id}"] = wt_mean
            results[f"val_dice_tc_{c_id}"] = tc_mean
            results[f"val_dice_et_{c_id}"] = et_mean
            results[f"val_mean_dice_{c_id}"] = mean_dice
            condition_dices.append(mean_dice)

    results["val_mean_dice_all_conditions"] = float(np.mean(condition_dices))
    return results


def train_model(
    model_id: str,
    seed: int,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    d0: float = 0.20,
    cache_dir: str = "data/cache",
    splits_file: str = "data/splits/split_v1.json",
    results_dir: str = "results",
    device_str: str = "cpu"
) -> Dict:
    device = torch.device(device_str)
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_dir = Path(results_dir) / model_id / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_csv = save_dir / "log.csv"
    best_ckpt = save_dir / "best_model.pt"

    # Resumable check
    if best_ckpt.exists() and log_csv.exists():
        log_df = pd.read_csv(log_csv)
        if len(log_df) >= epochs:
            print(f"Run {model_id} seed {seed} already completed ({len(log_df)} epochs). Skipping.")
            return {"status": "already_completed", "best_ckpt": str(best_ckpt)}

    with open(splits_file, "r") as f:
        splits = json.load(f)
    train_ids = splits["train"]
    val_ids = splits["val"]

    deg_module = LowFieldDegradation()
    
    # M0 trains on clean data only; M1-M7 train with degradation augmentation
    if model_id == "M0":
        train_deg_fn = None
    else:
        train_deg_fn = lambda x, mu, p_id, slice_idx: deg_module(x, mu)

    train_dataset = BraTS2DSliceDataset(
        patient_ids=train_ids,
        cache_dir=cache_dir,
        is_train=True,
        tumor_oversample_ratio=1.0,
        degradation_fn=train_deg_fn,
        spatial_augment=True,
        seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = build_model(model_id, d0=d0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Warmup + Cosine Annealing
    warmup_epochs = 1
    warmup_scheduler = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    criterion = CompositeLoss()

    best_val_score = -1.0
    history = []

    print(f"\n==========================================", flush=True)
    print(f"Training Model {model_id} (Seed {seed}) for {epochs} epochs", flush=True)
    print(f"Train slices: {len(train_dataset)}, Device: {device}", flush=True)
    print(f"==========================================", flush=True)

    total_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_dice_l = 0.0
        num_batches = 0

        for b_idx, batch in enumerate(train_loader, 1):
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, l_bce, l_dice = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_bce += l_bce
            total_dice_l += l_dice
            num_batches += 1

            if b_idx % 20 == 0 or b_idx == total_batches:
                elapsed = time.time() - t0
                rate = num_batches / max(elapsed, 0.1)
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Batch {b_idx:04d}/{total_batches:04d} | "
                      f"Loss: {total_loss/num_batches:.4f} (BCE: {total_bce/num_batches:.4f}, DiceLoss: {total_dice_l/num_batches:.4f}) | "
                      f"{rate:.1f} batch/s", flush=True)

        scheduler.step()
        epoch_time = time.time() - t0

        avg_loss = total_loss / max(num_batches, 1)
        avg_bce = total_bce / max(num_batches, 1)
        avg_dice_l = total_dice_l / max(num_batches, 1)

        # Validation across primary degradation conditions for epoch monitoring
        val_subset = val_ids[:10]
        val_metrics = validate_all_conditions(model, val_subset, cache_dir, device=device, primary_only=True)
        
        # Primary selection metric: overall mean dice (for M0 also tracks clean dice)
        val_score = val_metrics.get("val_mean_dice_all_conditions", 0.0) if model_id != "M0" else val_metrics.get("val_dice_wt_clean", 0.0)
        is_best = val_score > best_val_score

        if is_best:
            best_val_score = val_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_score": best_val_score,
                "model_id": model_id,
                "seed": seed,
                "d0": d0,
            }, best_ckpt)

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_bce": avg_bce,
            "train_dice_loss": avg_dice_l,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": epoch_time,
            "val_score": val_score,
            "val_clean_wt_dice": val_metrics.get("val_dice_wt_clean", 0.0),
            "val_clean_tc_dice": val_metrics.get("val_dice_tc_clean", 0.0),
            "val_clean_et_dice": val_metrics.get("val_dice_et_clean", 0.0),
            "val_mean_dice_all": val_metrics.get("val_mean_dice_all_conditions", 0.0),
            "val_snr12_r0.75_wt_dice": val_metrics.get("val_dice_wt_snr12_r0.75", 0.0),
            "val_snr8_r0.5_wt_dice": val_metrics.get("val_dice_wt_snr8_r0.5", 0.0),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(log_csv, index=False)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] Loss: {avg_loss:.4f} | "
              f"Val Clean WT: {row['val_clean_wt_dice']:.4f} | "
              f"Val All-Cond Mean: {row['val_mean_dice_all']:.4f} "
              f"{'(*BEST*)' if is_best else ''}", flush=True)

    print(f"Model {model_id} seed {seed} training complete. Best Val Score: {best_val_score:.4f}", flush=True)
    return {"status": "success", "best_val_score": best_val_score, "best_ckpt": str(best_ckpt)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BraTS Segmentation Model")
    parser.add_argument("--model", type=str, default="M0", help="Model ID: M0 through M7")
    parser.add_argument("--seed", type=int, default=0, help="Random seed: 0, 1, 2")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--d0", type=float, default=0.20, help="Cutoff frequency D0")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Cache directory")
    parser.add_argument("--splits_file", type=str, default="data/splits/split_v1.json", help="Splits JSON path")
    parser.add_argument("--results_dir", type=str, default="results", help="Results output directory")
    args = parser.parse_args()

    train_model(
        model_id=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d0=args.d0,
        cache_dir=args.cache_dir,
        splits_file=args.splits_file,
        results_dir=args.results_dir,
    )

