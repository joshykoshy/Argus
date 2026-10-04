"""
Data pipeline sanity check and visualization script.
Generates verification figure `figures/data_sanity.png` showcasing 3 patients x 4 modalities x GT overlays.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from python.data.manifest import build_manifest
from python.data.splits import create_stratified_split, load_splits
from python.data.preprocess import cache_all_patients
from python.data.dataset import BraTS2DSliceDataset, create_target_masks


def run_sanity_check(
    raw_dir: str,
    cache_dir: str,
    splits_file: str,
    manifest_file: str,
    output_fig: str
):
    print("=== Step 1: Building Manifest ===")
    mapping_csv = os.path.join(raw_dir, "name_mapping.csv")
    manifest_df = build_manifest(raw_dir, mapping_csv, manifest_file)
    assert len(manifest_df) == 369, f"Expected 369 patients, got {len(manifest_df)}"

    print("=== Step 2: Creating Stratified Split ===")
    if not os.path.exists(splits_file):
        splits = create_stratified_split(manifest_df, output_json=splits_file)
    else:
        splits = load_splits(splits_file)

    print("=== Step 3: Preprocessing and Caching All Volumes ===")
    retention_stats = cache_all_patients(manifest_df, cache_dir, force_recompute=True)

    print("=== Step 4: Generating Sanity Visualization Figure ===")
    val_patients = splits["val"][:3] # Pick first 3 validation patients
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9), constrained_layout=True)
    modality_names = ["T1", "T1ce", "T2", "FLAIR", "GT Overlay (WT/TC/ET)"]

    for row_idx, p_id in enumerate(val_patients):
        npz_path = os.path.join(cache_dir, f"{p_id}.npz")
        data = np.load(npz_path)
        modalities = data["modalities"] # (4, 192, 192, 155)
        seg = data["seg"]               # (192, 192, 155)

        # Find slice with maximum tumor area
        tumor_areas = [np.sum(seg[:, :, z] > 0) for z in range(seg.shape[2])]
        best_z = int(np.argmax(tumor_areas))
        if tumor_areas[best_z] == 0:
            best_z = seg.shape[2] // 2

        # 4 modalities
        for col_idx in range(4):
            img_slice = modalities[col_idx, :, :, best_z].astype(np.float32)
            axes[row_idx, col_idx].imshow(img_slice, cmap="gray", origin="lower")
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(modality_names[col_idx], fontsize=12, fontweight="bold")
            if col_idx == 0:
                grade = data["grade"]
                axes[row_idx, col_idx].text(
                    -20, 96, f"{p_id}\n({grade})\nz={best_z}",
                    va="center", ha="right", fontsize=10, fontweight="bold"
                )

        # GT Overlay on FLAIR:
        # WT: Red/Yellow outline, TC: Blue, ET: Green
        flair_slice = modalities[3, :, :, best_z].astype(np.float32)
        flair_norm = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)
        rgb_overlay = np.stack([flair_norm]*3, axis=-1)
        
        seg_sl = seg[:, :, best_z]
        # Label 1: NCR (Red: [1, 0, 0])
        # Label 2: ED (Green: [0, 1, 0])
        # Label 3: ET (Yellow: [1, 1, 0])
        rgb_overlay[seg_sl == 1] = [0.9, 0.2, 0.2] # NCR
        rgb_overlay[seg_sl == 2] = [0.2, 0.8, 0.2] # ED
        rgb_overlay[seg_sl == 3] = [1.0, 0.9, 0.1] # ET

        axes[row_idx, 4].imshow(rgb_overlay, origin="lower")
        axes[row_idx, 4].axis("off")
        if row_idx == 0:
            axes[row_idx, 4].set_title(modality_names[4], fontsize=12, fontweight="bold")

    Path(output_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sanity verification figure to {output_fig}")
    print("Gate G1 completed successfully!")


if __name__ == "__main__":
    raw_dir = r"C:\Users\Mayan\.cache\kagglehub\datasets\awsaf49\brats20-dataset-training-validation\versions\1\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    cache_dir = r"C:\Users\Mayan\.gemini\antigravity\scratch\ecte408\data\cache"
    splits_file = r"C:\Users\Mayan\.gemini\antigravity\scratch\ecte408\data\splits\split_v1.json"
    manifest_file = r"C:\Users\Mayan\.gemini\antigravity\scratch\ecte408\data\manifest.csv"
    output_fig = r"C:\Users\Mayan\.gemini\antigravity\scratch\ecte408\figures\data_sanity.png"
    run_sanity_check(raw_dir, cache_dir, splits_file, manifest_file, output_fig)
