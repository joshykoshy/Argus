"""
Unit tests for 3D Segmentation Metrics (Dice, HD95, Volume, and Edema-to-Core Ratio).
"""

import numpy as np
import pytest
from python.metrics.evaluation import (
    compute_3d_dice,
    compute_3d_hd95,
    compute_volume_cm3,
    evaluate_patient_3d
)


def test_toy_identical_and_disjoint_masks():
    shape = (100, 100, 100)
    
    # Identical mask
    mask_a = np.zeros(shape, dtype=np.uint8)
    mask_a[30:70, 30:70, 30:70] = 1
    
    dice_identical = compute_3d_dice(mask_a, mask_a)
    hd95_identical = compute_3d_hd95(mask_a, mask_a)
    
    assert abs(dice_identical - 1.0) < 1e-6, "Dice for identical masks must be 1.0"
    assert abs(hd95_identical - 0.0) < 1e-4, "HD95 for identical masks must be 0.0 mm"

    # Disjoint mask
    mask_b = np.zeros(shape, dtype=np.uint8)
    mask_b[0:20, 0:20, 0:20] = 1
    
    dice_disjoint = compute_3d_dice(mask_a, mask_b)
    assert abs(dice_disjoint - 0.0) < 1e-6, "Dice for disjoint masks must be 0.0"

    # Empty masks
    mask_empty = np.zeros(shape, dtype=np.uint8)
    assert compute_3d_dice(mask_empty, mask_empty) == 1.0, "Dice for both empty masks must be 1.0"
    assert compute_3d_dice(mask_a, mask_empty) == 0.0, "Dice for empty target must be 0.0"
    assert compute_3d_hd95(mask_a, mask_empty) == 373.13, "HD95 for empty mask must return BraTS penalty 373.13 mm"


def test_synthetic_sphere_volume():
    """
    Tests volume calculation on a synthetic 3D sphere of radius r=20 mm.
    Theoretical volume = 4/3 * pi * r^3 mm^3 = 4/3 * pi * 8000 mm^3 = 33510.32 mm^3 = 33.51 cm^3.
    """
    H, W, D = 100, 100, 100
    cz, cy, cx = 50, 50, 50
    radius = 20.0
    
    z, y, x = np.ogrid[:H, :W, :D]
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    sphere_mask = (dist_sq <= radius**2).astype(np.uint8)

    vol_computed_cm3 = compute_volume_cm3(sphere_mask, voxelspacing=(1.0, 1.0, 1.0))
    vol_theoretical_cm3 = (4.0 / 3.0) * np.pi * (radius**3) * 0.001

    rel_err = abs(vol_computed_cm3 - vol_theoretical_cm3) / vol_theoretical_cm3
    assert rel_err < 0.015, f"Sphere volume error {rel_err*100:.2f}% exceeds 1.5% discretization limit"
    print(f"Synthetic sphere volume: Computed={vol_computed_cm3:.3f} cm3, Theoretical={vol_theoretical_cm3:.3f} cm3 (error={rel_err*100:.2f}%)")
