"""
Exports representative test patient slices to .mat for fast MATLAB spectral analysis.
"""

import json
from pathlib import Path
import numpy as np
import scipy.io as sio

def export_test_slices_for_matlab(
    cache_dir: str = "data/cache",
    splits_file: str = "data/splits/split_v1.json",
    out_mat: str = "data/spectral_samples.mat",
    num_patients: int = 12
):
    with open(splits_file, "r") as f:
        splits = json.load(f)
    
    test_ids = splits["test"][:num_patients]
    slices_list = []
    masks_list = []
    mu_brains_list = []
    pids_list = []

    for pid in test_ids:
        npz_p = Path(cache_dir) / f"{pid}.npz"
        data = np.load(npz_p)
        modalities = data["modalities"] # (4, 192, 192, 155)
        seg = data["seg"]
        brain_mask = data["brain_mask"]
        mu_brain = data["mu_brain"]

        # Find slice with maximum tumor
        tumor_counts = [np.sum(seg[:, :, z] > 0) for z in range(seg.shape[2])]
        best_z = int(np.argmax(tumor_counts))
        if tumor_counts[best_z] == 0:
            best_z = seg.shape[2] // 2

        # Extract slice: (4, 192, 192)
        sl = modalities[:, :, :, best_z].astype(np.float32)
        m = brain_mask[:, :, best_z].astype(np.float32)

        slices_list.append(sl)
        masks_list.append(m)
        mu_brains_list.append(mu_brain.astype(np.float32))
        pids_list.append(pid)

    sio.savemat(out_mat, {
        "slices": np.stack(slices_list, axis=0),         # (N, 4, 192, 192)
        "masks": np.stack(masks_list, axis=0),           # (N, 192, 192)
        "mu_brains": np.stack(mu_brains_list, axis=0),   # (N, 4)
        "patient_ids": pids_list
    })
    print(f"Exported {num_patients} test slices to {out_mat}")

if __name__ == "__main__":
    export_test_slices_for_matlab()
