"""
Publication Figure Generator for ECTE408 Study.
Produces paper-ready figures in both PNG (300 DPI) and vector PDF formats:
1. Dice vs SNR (Main Figure across resolution levels and regions)
2. Dice Heatmaps (Model x Condition)
3. HD95 vs SNR
4. Volume Bland-Altman & Scatter plots
5. Qualitative Patient Comparisons
6. Training Convergence Curves
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


def plot_dice_vs_snr(raw_metrics_csv: str, out_prefix: str = "figures/dice_vs_snr"):
    df = pd.read_csv(raw_metrics_csv)
    
    # Exclude clean (snr=999) for continuous line plots, or plot separately
    deg_df = df[df["snr"] <= 30.0].copy()
    
    # Average across test patients and seeds
    summary = deg_df.groupby(["model", "snr", "r"], as_index=False)["dice_mean"].mean()

    models = sorted(summary["model"].unique())
    r_levels = [1.0, 0.75, 0.5]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    palette = sns.color_palette("tab10", len(models))

    for idx, r_val in enumerate(r_levels):
        ax = axes[idx]
        sub = summary[summary["r"] == r_val]
        for m_idx, m_name in enumerate(models):
            m_sub = sub[sub["model"] == m_name].sort_values("snr", ascending=False)
            ax.plot(
                m_sub["snr"], m_sub["dice_mean"],
                marker="o", linewidth=2.2, label=m_name, color=palette[m_idx]
            )
        ax.set_title(f"Resolution Retention: r = {r_val}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Signal-to-Noise Ratio (SNR, dB)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.invert_xaxis()
        if idx == 0:
            ax.set_ylabel("Mean 3D Dice Score", fontsize=11, fontweight="bold")
            ax.legend(title="Model", fontsize=9, title_fontsize=10)

    fig.suptitle("Brain Tumor Segmentation Robustness vs SNR Across k-Space Resolution Levels", fontsize=14, fontweight="bold")
    
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_prefix}.png and .pdf")


def plot_dice_heatmaps(raw_metrics_csv: str, out_prefix: str = "figures/dice_heatmaps"):
    df = pd.read_csv(raw_metrics_csv)
    summary = df.groupby(["model", "condition_id"], as_index=False)["dice_mean"].mean()
    
    pivot = summary.pivot(index="model", columns="condition_id", values="dice_mean")
    # Order columns logically
    col_order = ["clean", "snr30_r1.0", "snr30_r0.75", "snr30_r0.5",
                 "snr20_r1.0", "snr20_r0.75", "snr20_r0.5",
                 "snr12_r1.0", "snr12_r0.75", "snr12_r0.5",
                 "snr8_r1.0", "snr8_r0.75", "snr8_r0.5"]
    cols_present = [c for c in col_order if c in pivot.columns]
    pivot = pivot[cols_present]

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Mean 3D Dice Score"}, linewidths=0.5)
    plt.title("Mean 3D Dice Performance Across All 13 Degradation Conditions", fontsize=13, fontweight="bold")
    plt.xlabel("MRI Degradation Condition", fontsize=11, fontweight="bold")
    plt.ylabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_prefix}.png and .pdf")


def plot_hd95_vs_snr(raw_metrics_csv: str, out_prefix: str = "figures/hd95_vs_snr"):
    df = pd.read_csv(raw_metrics_csv)
    deg_df = df[df["snr"] <= 30.0].copy()
    summary = deg_df.groupby(["model", "snr"], as_index=False)["hd95_wt"].mean()

    plt.figure(figsize=(8, 5))
    models = sorted(summary["model"].unique())
    palette = sns.color_palette("tab10", len(models))

    for m_idx, m_name in enumerate(models):
        m_sub = summary[summary["model"] == m_name].sort_values("snr", ascending=False)
        plt.plot(m_sub["snr"], m_sub["hd95_wt"], marker="s", linewidth=2.0, label=m_name, color=palette[m_idx])

    plt.title("95th Percentile Hausdorff Distance (HD95) vs SNR (Whole Tumor)", fontsize=12, fontweight="bold")
    plt.xlabel("Signal-to-Noise Ratio (SNR, dB)", fontsize=11, fontweight="bold")
    plt.ylabel("HD95 (mm, lower is better)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().invert_xaxis()
    plt.legend(title="Model", fontsize=9)

    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_prefix}.png and .pdf")


def plot_volume_bland_altman(raw_metrics_csv: str, out_prefix: str = "figures/volume_bland_altman"):
    df = pd.read_csv(raw_metrics_csv)
    primary_models = ["M0", "M1", "M4"]
    target_cond = "snr12_r0.75"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for idx, m_name in enumerate(primary_models):
        ax = axes[idx]
        sub = df[(df["model"] == m_name) & (df["condition_id"] == target_cond)]
        
        gt_vol = sub["vol_gt_wt_cm3"].values
        pred_vol = sub["vol_pred_wt_cm3"].values

        mean_vol = (gt_vol + pred_vol) / 2.0
        diff_vol = pred_vol - gt_vol

        md = np.mean(diff_vol)
        sd = np.std(diff_vol)

        ax.scatter(mean_vol, diff_vol, alpha=0.6, color="#1f77b4", edgecolors="none", s=30)
        ax.axhline(md, color="black", linestyle="-", linewidth=1.5, label=f"Mean Bias: {md:+.2f}")
        ax.axhline(md + 1.96 * sd, color="red", linestyle="--", label=f"+1.96 SD: {md+1.96*sd:+.2f}")
        ax.axhline(md - 1.96 * sd, color="red", linestyle="--", label=f"-1.96 SD: {md-1.96*sd:+.2f}")

        ax.set_title(f"{m_name} (SNR 12, r=0.75)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean (Pred + GT) Volume (cm³)", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Difference (Pred - GT) Volume (cm³)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Bland-Altman Tumor Volume Agreement Under Low-Field Degradation", fontsize=13, fontweight="bold")

    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_prefix}.png and .pdf")


def plot_all_figures(raw_metrics_csv: str = "results/raw_metrics.csv"):
    plot_dice_vs_snr(raw_metrics_csv)
    plot_dice_heatmaps(raw_metrics_csv)
    plot_hd95_vs_snr(raw_metrics_csv)
    plot_volume_bland_altman(raw_metrics_csv)


if __name__ == "__main__":
    plot_all_figures()
