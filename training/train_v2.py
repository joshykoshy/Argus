"""
Training v2 (redesign Step A): fixes the problems the pilot diagnostic found.

What changed from python/train.py, and why
  slices      ALL brain slices of every training patient, tumor-free ones
              included. The v1 models saw only the 10 largest-tumor slices and
              drew tumor on 100 % of tumor-free slices (pilot/diagnose.py).
  validation  full-volume 3D Dice on all validation patients (all 155 slices),
              plus the false-tumor volume on tumor-free slices. v1 scored the 8
              largest-tumor slices only, which reported ~0.75 while true 3D Dice
              was ~0.2.
  loss        BCE + soft Dice summed over the whole BATCH per channel. v1's Dice
              was per slice, which is unstable on slices with no tumor.
  length      30 epochs (v1: 5), AdamW 3e-4, 1-epoch warm-up + cosine, mixed
              precision, checkpoint every epoch so a Colab disconnect resumes.
Unchanged: architectures (python.models.factory.build_model), preprocessing and
crop (python.data.preprocess), per-slice brain z-score normalisation, flips.

Degradation: none. Step A trains the clean baseline M0 only.
# ponytail: degradation augmentation (M1, M4, ...) is added in Step C, once the
# pilot has chosen the range; through-plane thickness needs neighbouring slices,
# so it belongs in the dataset, not the batch.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pilot.pilot import dice, normalise
from python.models.factory import build_model


class SliceSet(Dataset):
    """Every brain slice of the given patients, read from memory-mapped arrays."""

    def __init__(self, work, ids, flips):
        self.work, self.flips, self.x, self.y, self.m, self.index = Path(work), flips, {}, {}, {}, []
        for pid in ids:
            m = np.load(self.work / f"{pid}_m.npy", mmap_mode="r")
            self.index += [(pid, z) for z in np.nonzero(m.reshape(m.shape[0], -1).sum(1) > 100)[0]]

    def _arr(self, cache, pid, kind):
        if pid not in cache:                     # opened lazily: one handle per worker
            cache[pid] = np.load(self.work / f"{pid}_{kind}.npy", mmap_mode="r")
        return cache[pid]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pid, z = self.index[i]
        x = torch.from_numpy(np.array(self._arr(self.x, pid, "x")[z], dtype=np.float32))
        seg = torch.from_numpy(np.array(self._arr(self.y, pid, "y")[z]))
        m = torch.from_numpy(np.array(self._arr(self.m, pid, "m")[z]))
        y = torch.stack([seg > 0, (seg == 1) | (seg == 3), seg == 3]).float()   # WT, TC, ET
        if self.flips:
            for dim in (-1, -2):
                if random.random() < 0.5:
                    x, y, m = x.flip(dim), y.flip(dim), m.flip(dim)
        return x, y, m


def loss_fn(p, t, eps=1.0):
    """BCE + soft Dice, Dice summed over the whole batch per channel."""
    bce = torch.nn.functional.binary_cross_entropy(p, t)
    inter = (p * t).sum(dim=(0, 2, 3))
    denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    return bce + (1 - (2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def validate(model, work, ids, device):
    """Full-volume 3D Dice per patient (mean of WT, TC, ET) + false tumor on tumor-free slices."""
    model.eval()
    d3, fp_cm3 = [], []
    for pid in ids:
        x = torch.from_numpy(np.load(Path(work) / f"{pid}_x.npy").astype(np.float32)).to(device)
        m = torch.from_numpy(np.load(Path(work) / f"{pid}_m.npy")).to(device)
        seg = torch.from_numpy(np.load(Path(work) / f"{pid}_y.npy")).to(device)
        xn = normalise(x, m)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.cat([model(xn[s:s + 32]) for s in range(0, xn.shape[0], 32)]).float() > 0.5
        gts = [seg > 0, (seg == 1) | (seg == 3), seg == 3]
        d3.append(np.mean([dice(p[:, k], gts[k]) for k in range(3)]))
        free = gts[0].sum(dim=(1, 2)) == 0
        fp_cm3.append(p[free, 0].sum().item() / 1000)      # 1 mm^3 voxels
    model.train()
    return float(np.mean(d3)), float(np.mean(fp_cm3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", required=True, help="folder written by training.prepare")
    ap.add_argument("--out", required=True, help="checkpoints + log (put it on Drive)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--slices_per_epoch", type=int, default=12000,
                    help="random brain slices drawn per epoch (0 = all)")
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--val_patients", type=int, default=0, help="0 = all validation patients")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out) / args.model / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    split = json.load(open("data/splits/split_v1.json"))
    val_ids = split["val"][: args.val_patients] if args.val_patients else split["val"]
    data = SliceSet(args.work, split["train"], flips=True)
    n_tumor = sum(1 for pid, z in data.index[:5000] if np.load(Path(args.work) / f"{pid}_y.npy", mmap_mode="r")[z].any())
    print(f"{len(data)} brain slices from {len(split['train'])} patients "
          f"(~{100 * n_tumor / min(5000, len(data)):.0f}% contain tumor); device {device}", flush=True)

    model = build_model(args.model, d0=0.20).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=1)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs - 1, 1), eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[1])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start, best = 1, -1.0

    last = out / "last.pt"
    if last.exists():                           # resume after a disconnect
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"Resuming from epoch {start} (best 3D Dice so far {best:.4f})", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    log = out / "log.csv"
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        n = args.slices_per_epoch or len(data)
        sampler = torch.utils.data.RandomSampler(data, num_samples=min(n, len(data)), generator=g)
        loader = DataLoader(data, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                            pin_memory=True, drop_last=True, persistent_workers=False)
        model.train()
        tot, nb = 0.0, 0
        for x, y, m in loader:
            x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device)
            xn = normalise(x, m)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                p = model(xn)
            loss = loss_fn(p.float().clamp(1e-6, 1 - 1e-6), y)   # loss in fp32 (BCE is not autocast-safe)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot, nb = tot + loss.item(), nb + 1
            if nb % 100 == 0:
                print(f"  epoch {epoch} batch {nb}/{len(loader)} loss {tot / nb:.4f} "
                      f"({nb / (time.time() - t0):.1f} batch/s)", flush=True)
        sched.step()

        row = dict(epoch=epoch, train_loss=tot / max(nb, 1), lr=opt.param_groups[0]["lr"],
                   epoch_s=round(time.time() - t0), val_dice_3d="", val_false_tumor_cm3="")
        if epoch % args.val_every == 0 or epoch == args.epochs:
            d3, fp = validate(model, args.work, val_ids, device)
            row.update(val_dice_3d=round(d3, 4), val_false_tumor_cm3=round(fp, 1))
            if d3 > best:
                best = d3
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_dice_3d": d3},
                           out / "best_model.pt")
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "best": best}, last)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f}, val 3D Dice {row['val_dice_3d']}, "
              f"false tumor {row['val_false_tumor_cm3']} cm3, {row['epoch_s']} s", flush=True)
    print(f"TRAINING DONE. Best full-volume 3D Dice {best:.4f} -> {out / 'best_model.pt'}")


if __name__ == "__main__":
    main()
