"""
Evaluates trained M0 model across all 13 MRI degradation conditions and generates `figures/m0_degradation_curve.png`.
Verifies Gate G4 (Clean WT Dice >= 0.80 and quantitative degradation curve).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from python.data.dataset import BraTS2DSliceDataset
from python.physics.degradation import LowFieldDegradation, DEGRADATION_CONDITIONS
from python.models.factory import build_model
from python.metrics.evaluation import compute_3d_dice


def generate_m0_degradation_curve(
    checkpoint_path="results/M0/seed_0/best_model.pt",
    splits_file="data/splits/split_v1.json",
    cache_dir="data/cache",
    out_fig="figures/m0_degradation_curve.png",
    out_csv="results/m0_degradation_metrics.csv"
):
    with open(splits_file) as f:
        splits = json.load(f)
    val_ids = splits["val"][:15]  # 15 validation patients

    device = torch.device("cpu")
    model = build_model("M0").to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    deg = LowFieldDegradation()
    records = []

    print("Evaluating M0 across all 13 degradation conditions...")

    for cond in DEGRADATION_CONDITIONS:
        c_id = cond["id"]
        snr = cond["snr"]
        r = cond["r"]

        val_ds = BraTS2DSliceDataset(
            patient_ids=val_ids,
            cache_dir=cache_dir,
            is_train=False,
            degradation_fn=lambda x, mu, p_id, slice_idx, _c=c_id: deg(
                x, mu, condition_id=_c, patient_id=p_id, slice_idx=slice_idx
            ),
            spatial_augment=False,
        )
        loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        wt_dices, tc_dices, et_dices = [], [], []

        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                tgts = batch["target"].to(device)
                preds = model(imgs)

                for b in range(imgs.shape[0]):
                    p_np = preds[b].cpu().numpy()
                    t_np = tgts[b].cpu().numpy()

                    wt_dices.append(compute_3d_dice(p_np[0], t_np[0]))
                    tc_dices.append(compute_3d_dice(p_np[1], t_np[1]))
                    et_dices.append(compute_3d_dice(p_np[2], t_np[2]))

        mean_wt = float(np.mean(wt_dices))
        mean_tc = float(np.mean(tc_dices))
        mean_et = float(np.mean(et_dices))
        mean_all = (mean_wt + mean_tc + mean_et) / 3.0

        records.append({
            "condition_id": c_id,
            "snr": snr if snr is not None else 999.0,
            "r": r,
            "dice_wt": round(mean_wt, 4),
            "dice_tc": round(mean_tc, 4),
            "dice_et": round(mean_et, 4),
            "dice_mean": round(mean_all, 4),
        })
        print(f"  Condition: {c_id:<15} | WT Dice: {mean_wt:.4f} | TC Dice: {mean_tc:.4f} | ET Dice: {mean_et:.4f} | Mean: {mean_all:.4f}")

    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved degradation metrics to {out_csv}")

    # Plot degradation curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    r_vals = [1.0, 0.75, 0.5]
    colors = {"WT": "#1f77b4", "TC": "#ff7f0e", "ET": "#2ca02c", "Mean": "#d62728"}

    for idx, r_val in enumerate(r_vals):
        ax = axes[idx]
        sub = df[(df["r"] == r_val) & (df["snr"] <= 30.0)].sort_values("snr", ascending=False)
        clean_row = df[df["condition_id"] == "clean"].iloc[0]

        ax.plot(sub["snr"], sub["dice_wt"], "o-", color=colors["WT"], label="WT (Whole Tumor)", lw=2)
        ax.plot(sub["snr"], sub["dice_tc"], "s-", color=colors["TC"], label="TC (Tumor Core)", lw=2)
        ax.plot(sub["snr"], sub["dice_et"], "^-", color=colors["ET"], label="ET (Enhancing)", lw=2)
        ax.plot(sub["snr"], sub["dice_mean"], "D--", color=colors["Mean"], label="Mean Tumor", lw=2.5)

        # Baseline clean reference
        ax.axhline(clean_row["dice_mean"], color="gray", ls=":", alpha=0.7, label="Clean Baseline (Mean)")

        ax.set_title(f"Resolution Retention: r = {r_val}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Signal-to-Noise Ratio (SNR, dB)", fontsize=11, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.5)
        ax.invert_xaxis()
        if idx == 0:
            ax.set_ylabel("Validation Dice Score", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9, loc="lower left")

    fig.suptitle("M0 Baseline Clean U-Net: Severe Degradation Breakdown (Hypothesis H1)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved M0 degradation curve figure to {out_fig}")
    return df


if __name__ == "__main__":
    generate_m0_degradation_curve()
