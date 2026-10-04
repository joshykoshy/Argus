"""
Full 3D Test-Set Evaluation Engine for ECTE408 Study.
Reconstructs 3D patient volumes from 2D slice predictions.
Evaluates all models, seeds, conditions, and test patients on Dice, HD95, Volume, and Edema Ratio.
Saves `results/raw_metrics.csv` and predicted NIfTI masks for primary conditions.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from tqdm import tqdm

from python.data.dataset import BraTS2DSliceDataset, reconstruct_3d_volume
from python.physics.degradation import LowFieldDegradation, DEGRADATION_CONDITIONS
from python.models.factory import build_model
from python.metrics.evaluation import evaluate_patient_3d


def load_trained_model(model_id: str, checkpoint_path: str, device: torch.device = torch.device("cpu"), d0: float = 0.20) -> nn.Module:
    model = build_model(model_id, d0=d0).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_patient_volume(
    model: nn.Module,
    patient_id: str,
    condition_id: str,
    cache_dir: str,
    deg_module: LowFieldDegradation,
    device: torch.device = torch.device("cpu")
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, Tuple]:
    """
    Evaluates a single patient volume under a specific degradation condition.
    Returns computed 3D metrics dictionary, 3D binary predictions (3, 240, 240, 155),
    3D binary ground truth (3, 240, 240, 155), affine matrix, and zooms.
    """
    npz_path = Path(cache_dir) / f"{patient_id}.npz"
    data = np.load(npz_path)

    modalities = data["modalities"] # (4, 192, 192, 155) float16
    seg = data["seg"]               # (192, 192, 155) uint8
    brain_mask = data["brain_mask"] # (192, 192, 155) uint8
    mu_brain = data["mu_brain"]     # (4,) float32
    affine = data["affine"]
    zooms = tuple(float(z) for z in data["zooms"])
    crop_coords = tuple(data["crop_coords"])
    orig_shape = tuple(data["orig_shape"])
    grade = str(data["grade"])

    num_slices = modalities.shape[3] # 155
    slice_preds = []

    # Process slice by slice
    for z in range(num_slices):
        x_2d = torch.from_numpy(modalities[:, :, :, z].astype(np.float32)) # (4, 192, 192)
        m_2d = torch.from_numpy(brain_mask[:, :, :, z].astype(np.float32)) if brain_mask.ndim == 4 else torch.from_numpy(brain_mask[:, :, z].astype(np.float32))
        mu_b = torch.from_numpy(mu_brain)

        # Apply deterministic degradation
        x_deg, _ = deg_module(
            x_2d, mu_b, condition_id=condition_id, patient_id=patient_id, slice_idx=z
        )

        # Normalization inside brain mask
        x_norm = torch.zeros_like(x_deg)
        mask_bool = m_2d > 0.5
        for c in range(4):
            ch = x_deg[c]
            if mask_bool.sum() > 0:
                vals = ch[mask_bool]
                mean_v = vals.mean()
                std_v = vals.std()
                if std_v < 1e-6:
                    std_v = 1.0
                x_norm[c] = (ch - mean_v) / std_v
            else:
                x_norm[c] = ch

        # Inference
        with torch.no_grad():
            inp = x_norm.unsqueeze(0).to(device)
            p = model(inp).squeeze(0).cpu().numpy() # (3, 192, 192)
            slice_preds.append(p)

    slice_preds_np = np.stack(slice_preds, axis=0) # (155, 3, 192, 192)

    # Reconstruct full 3D prediction volume: (3, 240, 240, 155)
    pred_3d_prob = reconstruct_3d_volume(slice_preds_np, crop_coords=crop_coords, orig_shape=orig_shape)
    pred_3d_bin = (pred_3d_prob > 0.5).astype(np.float32)

    # Build 3D ground truth: (3, 240, 240, 155)
    gt_crop_wt = (seg > 0).astype(np.float32)
    gt_crop_tc = ((seg == 1) | (seg == 3)).astype(np.float32)
    gt_crop_et = (seg == 3).astype(np.float32)
    gt_crop_3d = np.stack([gt_crop_wt, gt_crop_tc, gt_crop_et], axis=0) # (3, 192, 192, 155)
    
    # Reconstruct full GT volume
    gt_3d = np.zeros((3, orig_shape[0], orig_shape[1], orig_shape[2]), dtype=np.float32)
    r_s, r_e, c_s, c_e = crop_coords
    gt_3d[:, r_s:r_e, c_s:c_e, :] = gt_crop_3d

    # Evaluate 3D metrics
    metrics = evaluate_patient_3d(pred_3d_bin, gt_3d, voxelspacing=zooms)
    metrics["patient_id"] = patient_id
    metrics["grade"] = grade
    metrics["condition_id"] = condition_id

    return metrics, pred_3d_bin, gt_3d, affine, zooms


def run_full_test_evaluation(
    model_ids: List[str] = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"],
    seeds: List[int] = [0, 1, 2],
    splits_file: str = "data/splits/split_v1.json",
    cache_dir: str = "data/cache",
    results_dir: str = "results",
    d0: float = 0.20,
    device_str: str = "cpu"
) -> pd.DataFrame:
    device = torch.device(device_str)
    deg_module = LowFieldDegradation()

    with open(splits_file, "r") as f:
        splits = json.load(f)
    test_ids = splits["test"]

    print(f"Starting Full Test-Set Evaluation on {len(test_ids)} patients across all conditions...")

    all_records = []
    primary_conditions = ["clean", "snr12_r0.75", "snr8_r0.5"]
    nii_save_dir = Path(results_dir) / "predictions"
    nii_save_dir.mkdir(parents=True, exist_ok=True)

    for m_id in model_ids:
        for seed in seeds:
            ckpt_path = str(Path(results_dir) / m_id / f"seed_{seed}" / "best_model.pt")
            if not os.path.exists(ckpt_path):
                print(f"Skipping {m_id} seed {seed} (checkpoint not found at {ckpt_path})")
                continue
            model = load_trained_model(m_id, ckpt_path, device=device, d0=d0)

            for cond in DEGRADATION_CONDITIONS:
                c_id = cond["id"]
                snr_val = cond["snr"] if cond["snr"] is not None else 999.0
                r_val = cond["r"]

                for p_id in tqdm(test_ids, desc=f"Eval {m_id} s{seed} {c_id}", leave=False):
                    metrics, pred_3d_bin, gt_3d, affine, zooms = evaluate_patient_volume(
                        model, p_id, c_id, cache_dir, deg_module, device=device
                    )

                    # Save NIfTI mask for primary conditions on seed 0
                    if seed == 0 and c_id in primary_conditions:
                        # Combine 3 channels into integer segmentation (1=NCR, 2=ED, 3=ET)
                        # WT=1,2,3; TC=1,3; ET=3
                        # NCR = TC & ~ET -> 1
                        # ET = ET -> 3 (or 4 in BraTS format)
                        # ED = WT & ~TC -> 2
                        pred_int = np.zeros(pred_3d_bin.shape[1:], dtype=np.uint8)
                        wt_m = pred_3d_bin[0] > 0.5
                        tc_m = pred_3d_bin[1] > 0.5
                        et_m = pred_3d_bin[2] > 0.5

                        pred_int[wt_m] = 2 # ED default
                        pred_int[tc_m] = 1 # NCR
                        pred_int[et_m] = 3 # ET (remapped 3)

                        out_nii_path = nii_save_dir / f"{m_id}_{p_id}_{c_id}.nii.gz"
                        nii_obj = nib.Nifti1Image(pred_int, affine)
                        nib.save(nii_obj, str(out_nii_path))

                    rec = {
                        "model": m_id,
                        "seed": seed,
                        "patient_id": p_id,
                        "grade": metrics["grade"],
                        "condition_id": c_id,
                        "snr": snr_val,
                        "r": r_val,
                        "dice_wt": metrics["dice_wt"],
                        "dice_tc": metrics["dice_tc"],
                        "dice_et": metrics["dice_et"],
                        "dice_mean": (metrics["dice_wt"] + metrics["dice_tc"] + metrics["dice_et"]) / 3.0,
                        "hd95_wt": metrics["hd95_wt"],
                        "hd95_tc": metrics["hd95_tc"],
                        "hd95_et": metrics["hd95_et"],
                        "vol_pred_wt_cm3": metrics["vol_pred_wt_cm3"],
                        "vol_gt_wt_cm3": metrics["vol_gt_wt_cm3"],
                        "vol_abs_err_wt_cm3": metrics["vol_abs_err_wt_cm3"],
                        "vol_rel_err_wt": metrics["vol_rel_err_wt"],
                        "vol_pred_tc_cm3": metrics["vol_pred_tc_cm3"],
                        "vol_gt_tc_cm3": metrics["vol_gt_tc_cm3"],
                        "vol_pred_et_cm3": metrics["vol_pred_et_cm3"],
                        "vol_gt_et_cm3": metrics["vol_gt_et_cm3"],
                        "edema_core_ratio_pred": metrics["edema_core_ratio_pred"],
                        "edema_core_ratio_gt": metrics["edema_core_ratio_gt"],
                        "edema_core_ratio_abs_err": metrics["edema_core_ratio_abs_err"],
                    }
                    all_records.append(rec)

    # Save Ground Truth NIfTI masks for primary conditions as reference for MATLAB volumetrics
    gt_nii_dir = nii_save_dir / "ground_truth"
    gt_nii_dir.mkdir(parents=True, exist_ok=True)
    for p_id in test_ids:
        npz_p = Path(cache_dir) / f"{p_id}.npz"
        data = np.load(npz_p)
        seg_cr = data["seg"] # (192, 192, 155)
        affine = data["affine"]
        crop_coords = tuple(data["crop_coords"])
        orig_shape = tuple(data["orig_shape"])

        gt_full = np.zeros(orig_shape, dtype=np.uint8)
        r_s, r_e, c_s, c_e = crop_coords
        gt_full[r_s:r_e, c_s:c_e, :] = seg_cr

        gt_nii_file = gt_nii_dir / f"GT_{p_id}.nii.gz"
        nib.save(nib.Nifti1Image(gt_full, affine), str(gt_nii_file))

    raw_metrics_df = pd.DataFrame(all_records)
    out_csv = Path(results_dir) / "raw_metrics.csv"
    raw_metrics_df.to_csv(out_csv, index=False)
    print(f"Saved {len(raw_metrics_df)} full test evaluation records to {out_csv}")
    return raw_metrics_df
