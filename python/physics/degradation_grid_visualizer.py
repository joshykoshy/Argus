"""
Degradation Grid Visualizer.
Generates `figures/degradation_grid.png` illustrating 1 sample slice across 4 modalities and all 13 degradation conditions.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from python.physics.degradation import (
    apply_degradation_single_slice,
    DEGRADATION_CONDITIONS,
    CONDITION_MAP
)


def generate_degradation_grid_figure(
    cache_dir: str,
    patient_id: str,
    output_path: str
):
    npz_file = Path(cache_dir) / f"{patient_id}.npz"
    if not npz_file.exists():
        raise FileNotFoundError(f"Missing cached file: {npz_file}")

    data = np.load(npz_file)
    modalities = data["modalities"] # (4, 192, 192, 155)
    seg = data["seg"]               # (192, 192, 155)
    mu_brain = data["mu_brain"]     # (4,)

    # Find slice with prominent tumor
    tumor_counts = [np.sum(seg[:, :, z] > 0) for z in range(seg.shape[2])]
    best_z = int(np.argmax(tumor_counts))

    slice_clean = torch.from_numpy(modalities[:, :, :, best_z].astype(np.float32)) # (4, 192, 192)
    mu_b = torch.from_numpy(mu_brain)

    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    num_conds = len(DEGRADATION_CONDITIONS) # 13

    fig, axes = plt.subplots(num_conds, 4, figsize=(12, 32), constrained_layout=True)

    for r_idx, cond in enumerate(DEGRADATION_CONDITIONS):
        c_id = cond["id"]
        snr = cond["snr"]
        r_val = cond["r"]

        deg_slice = apply_degradation_single_slice(
            slice_clean, mu_b, snr=snr, r=r_val, patient_id=patient_id, slice_idx=best_z, condition_id=c_id
        ).numpy()

        for c_idx in range(4):
            ax = axes[r_idx, c_idx]
            img = deg_slice[c_idx]
            ax.imshow(img, cmap="gray", origin="lower")
            ax.axis("off")

            if r_idx == 0:
                ax.set_title(modality_names[c_idx], fontsize=13, fontweight="bold")
            if c_idx == 0:
                if snr is None:
                    label_str = "Clean\n(r=1.0)"
                else:
                    label_str = f"SNR={int(snr)}\nr={r_val}"
                ax.text(-15, 96, label_str, va="center", ha="right", fontsize=11, fontweight="bold")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved degradation grid figure to {output_path}")


if __name__ == "__main__":
    cache_dir = r"C:\Users\Mayan\.gemini\antigravity\scratch\ecte408\data\cache"
    splits_file = r"C:\Users\Mayan\.gemini\antigravity\scratch\ecte408\data\splits\split_v1.json"
    import json
    with open(splits_file, "r") as f:
        splits = json.load(f)
    sample_p = splits["val"][0]
    out_fig = r"C:\Users\Mayan\.gemini\antigravity\scratch\ecte408\figures\degradation_grid.png"
    generate_degradation_grid_figure(cache_dir, sample_p, out_fig)
