"""
Whole-Volume 3D Evaluation Metrics.
Computes 3D Dice, 95th Percentile Hausdorff Distance (HD95, mm),
Volumetrics (cm3), and Edema-to-Core ratio with BraTS 2020 penalty rules.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from medpy.metric.binary import hd95


def compute_3d_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    Computes 3D binary Dice coefficient for a single region.
    If target is empty: Dice = 1.0 if pred is empty, else 0.0.
    """
    pred_b = (pred > 0.5).astype(bool)
    target_b = (target > 0.5).astype(bool)

    target_sum = np.sum(target_b)
    pred_sum = np.sum(pred_b)

    if target_sum == 0:
        return 1.0 if pred_sum == 0 else 0.0

    intersection = np.sum(pred_b & target_b)
    return float((2.0 * intersection) / (pred_sum + target_sum + eps))


def compute_3d_hd95(pred: np.ndarray, target: np.ndarray, voxelspacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """
    Computes 95th percentile Hausdorff Distance in mm.
    If either mask is empty, returns standard BraTS penalty: 373.13 mm.
    """
    pred_b = (pred > 0.5).astype(bool)
    target_b = (target > 0.5).astype(bool)

    if np.sum(pred_b) == 0 or np.sum(target_b) == 0:
        return 373.13

    try:
        union = pred_b | target_b
        coords = np.argwhere(union)
        rmin, cmin, zmin = coords.min(axis=0)
        rmax, cmax, zmax = coords.max(axis=0) + 1
        dist = hd95(pred_b[rmin:rmax, cmin:cmax, zmin:zmax], target_b[rmin:rmax, cmin:cmax, zmin:zmax], voxelspacing=voxelspacing)
        return float(dist)
    except Exception:
        return 373.13


def compute_volume_cm3(mask: np.ndarray, voxelspacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Computes physical volume in cm3 (1 mm3 = 0.001 cm3)."""
    mask_b = (mask > 0.5).astype(bool)
    voxel_vol_mm3 = float(voxelspacing[0] * voxelspacing[1] * voxelspacing[2])
    voxel_count = np.sum(mask_b)
    return float(voxel_count * voxel_vol_mm3 * 0.001)


def evaluate_patient_3d(
    pred_3d: np.ndarray,     # (3, 240, 240, 155) -> WT, TC, ET
    target_3d: np.ndarray,   # (3, 240, 240, 155) -> WT, TC, ET
    voxelspacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, float]:
    """
    Evaluates 3D metrics across WT, TC, ET regions.
    """
    region_names = ["WT", "TC", "ET"]
    metrics = {}

    for i, r_name in enumerate(region_names):
        p_ch = pred_3d[i]
        t_ch = target_3d[i]

        d = compute_3d_dice(p_ch, t_ch)
        h = compute_3d_hd95(p_ch, t_ch, voxelspacing=voxelspacing)
        vol_p = compute_volume_cm3(p_ch, voxelspacing=voxelspacing)
        vol_t = compute_volume_cm3(t_ch, voxelspacing=voxelspacing)

        abs_err = abs(vol_p - vol_t)
        rel_err = abs_err / max(vol_t, 1e-3)
        signed_err = vol_p - vol_t

        metrics[f"dice_{r_name.lower()}"] = d
        metrics[f"hd95_{r_name.lower()}"] = h
        metrics[f"vol_pred_{r_name.lower()}_cm3"] = vol_p
        metrics[f"vol_gt_{r_name.lower()}_cm3"] = vol_t
        metrics[f"vol_abs_err_{r_name.lower()}_cm3"] = abs_err
        metrics[f"vol_rel_err_{r_name.lower()}"] = rel_err
        metrics[f"vol_signed_err_{r_name.lower()}_cm3"] = signed_err

    # Edema-to-core ratio
    vol_wt_p = metrics["vol_pred_wt_cm3"]
    vol_tc_p = metrics["vol_pred_tc_cm3"]
    vol_ed_p = max(0.0, vol_wt_p - vol_tc_p)
    ratio_p = vol_ed_p / max(vol_tc_p, 1e-3)

    vol_wt_t = metrics["vol_gt_wt_cm3"]
    vol_tc_t = metrics["vol_gt_tc_cm3"]
    vol_ed_t = max(0.0, vol_wt_t - vol_tc_t)
    ratio_t = vol_ed_t / max(vol_tc_t, 1e-3)

    metrics["edema_core_ratio_pred"] = ratio_p
    metrics["edema_core_ratio_gt"] = ratio_t
    metrics["edema_core_ratio_abs_err"] = abs(ratio_p - ratio_t)

    return metrics
