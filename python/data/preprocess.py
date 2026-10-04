"""
Volume preprocessing and caching module for BraTS 2020.
Handles NIfTI loading, label remapping (4->3), brain mask computation,
in-plane 192x192 cropping, affine/spacing preservation, and float16 disk caching.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_crop_indices(orig_shape: Tuple[int, int] = (240, 240), crop_size: Tuple[int, int] = (192, 192)) -> Tuple[int, int, int, int]:
    """
    Computes optimal 192x192 crop coordinates [r_start:r_end, c_start:c_end]
    derived from training-set brain bounding envelope.
    """
    r_start = 24
    r_end = r_start + crop_size[0] # 216
    c_start = 28
    c_end = c_start + crop_size[1] # 220
    return r_start, r_end, c_start, c_end


def preprocess_patient(row: pd.Series, crop_size: Tuple[int, int] = (192, 192)) -> Dict:
    """
    Loads 4 modalities and segmentation for one patient, applies centered cropping,
    remaps segmentation labels (4 -> 3), and computes the brain mask.
    """
    t1_img = nib.load(row["t1_path"])
    t1ce_img = nib.load(row["t1ce_path"])
    t2_img = nib.load(row["t2_path"])
    flair_img = nib.load(row["flair_path"])
    seg_img = nib.load(row["seg_path"])

    affine = seg_img.affine
    header = seg_img.header
    zooms = header.get_zooms()[:3]  # Voxel spacing (dx, dy, dz)

    t1_data = t1_img.get_fdata(dtype=np.float32)
    t1ce_data = t1ce_img.get_fdata(dtype=np.float32)
    t2_data = t2_img.get_fdata(dtype=np.float32)
    flair_data = flair_img.get_fdata(dtype=np.float32)
    seg_data = np.asanyarray(seg_img.dataobj, dtype=np.uint8)

    # Stack modalities: (4, H, W, D) -> (4, 240, 240, 155)
    modalities = np.stack([t1_data, t1ce_data, t2_data, flair_data], axis=0)

    # Brain mask: nonzero voxels across all modalities (skull-stripped)
    brain_mask = (np.abs(modalities).sum(axis=0) > 1e-4).astype(np.uint8)

    # Remap label 4 -> 3 (1=NCR/NET, 2=ED, 3=ET)
    seg_remapped = np.zeros_like(seg_data, dtype=np.uint8)
    seg_remapped[seg_data == 1] = 1
    seg_remapped[seg_data == 2] = 2
    seg_remapped[seg_data == 4] = 3

    # In-plane crop from 240x240 to 192x192
    r_s, r_e, c_s, c_e = compute_crop_indices((240, 240), crop_size)
    
    # Check brain mask retention
    total_brain_voxels = np.sum(brain_mask)
    cropped_brain_mask = brain_mask[r_s:r_e, c_s:c_e, :]
    retained_brain_voxels = np.sum(cropped_brain_mask)
    retention_ratio = retained_brain_voxels / max(total_brain_voxels, 1)

    cropped_modalities = modalities[:, r_s:r_e, c_s:c_e, :]
    cropped_seg = seg_remapped[r_s:r_e, c_s:c_e, :]

    # Per-modality, per-patient brain mean intensity for clean Rician noise calibration
    # mu_brain[c] = mean clean intensity inside brain mask for modality c
    mu_brain = np.zeros(4, dtype=np.float32)
    for c in range(4):
        vals = cropped_modalities[c][cropped_brain_mask > 0]
        mu_brain[c] = float(np.mean(vals)) if len(vals) > 0 else 1.0

    return {
        "patient_id": row["patient_id"],
        "grade": row["grade"],
        "modalities": cropped_modalities.astype(np.float16),  # (4, 192, 192, 155)
        "seg": cropped_seg,                                  # (192, 192, 155), uint8
        "brain_mask": cropped_brain_mask,                    # (192, 192, 155), uint8
        "mu_brain": mu_brain,                                # (4,) float32
        "affine": affine,                                    # 4x4
        "zooms": np.array(zooms, dtype=np.float32),          # (3,)
        "orig_shape": (240, 240, 155),
        "crop_coords": (r_s, r_e, c_s, c_e),
        "retention_ratio": retention_ratio,
    }


from concurrent.futures import ProcessPoolExecutor, as_completed


def _process_and_save_single_patient(args):
    row_dict, cache_dir, crop_size, force_recompute = args
    row = pd.Series(row_dict)
    p_id = row["patient_id"]
    out_file = Path(cache_dir) / f"{p_id}.npz"

    if out_file.exists() and not force_recompute:
        data = np.load(out_file)
        ret_ratio = float(data["retention_ratio"])
        return p_id, ret_ratio

    processed = preprocess_patient(row, crop_size)
    np.savez_compressed(
        out_file,
        modalities=processed["modalities"],
        seg=processed["seg"],
        brain_mask=processed["brain_mask"],
        mu_brain=processed["mu_brain"],
        affine=processed["affine"],
        zooms=processed["zooms"],
        crop_coords=np.array(processed["crop_coords"]),
        orig_shape=np.array(processed["orig_shape"]),
        grade=processed["grade"],
        retention_ratio=processed["retention_ratio"]
    )
    return p_id, processed["retention_ratio"]


def cache_all_patients(
    manifest_df: pd.DataFrame,
    cache_dir: str,
    crop_size: Tuple[int, int] = (192, 192),
    force_recompute: bool = False,
    num_workers: int = 8
) -> Dict[str, float]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    retention_stats = {}
    logger.info(f"Preprocessing and caching {len(manifest_df)} patients into {cache_dir} with {num_workers} workers...")

    tasks = [
        (row.to_dict(), cache_dir, crop_size, force_recompute)
        for _, row in manifest_df.iterrows()
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_and_save_single_patient, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Caching BraTS 2020 (Parallel)"):
            p_id, ret_ratio = f.result()
            retention_stats[p_id] = ret_ratio

    if retention_stats:
        min_ret = min(retention_stats.values())
        mean_ret = np.mean(list(retention_stats.values()))
        logger.info(f"Brain mask voxel retention: mean={mean_ret*100:.3f}%, min={min_ret*100:.3f}%")
        assert min_ret >= 0.999, f"Crop retention failed requirement: min retention {min_ret*100:.3f}% < 99.9%"

    logger.info("Caching complete.")
    return retention_stats
