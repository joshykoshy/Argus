"""Light unit tests for evaluation/evaluate_v2.py (tiny arrays only; no models, no BraTS data)."""

import numpy as np
import nibabel as nib
import pandas as pd

from evaluation.evaluate_v2 import (COLUMNS, HD95_EMPTY_PENALTY, append_rows, completed_pairs, hd95_brats,
                                    label_volume, n_components, save_nifti, to_full_grid)
from python.data.preprocess import compute_crop_indices


def ball(shape, c, r):
    g = np.indices(shape)
    return sum((g[i] - c[i]) ** 2 for i in range(3)) < r ** 2


def test_hd95_both_empty_is_zero():
    z = np.zeros((20, 20, 20), bool)
    assert hd95_brats(z, z, (1, 1, 1)) == 0.0


def test_hd95_one_empty_is_penalty():
    z, b = np.zeros((20, 20, 20), bool), ball((20, 20, 20), (10, 10, 10), 4)
    assert hd95_brats(z, b, (1, 1, 1)) == HD95_EMPTY_PENALTY
    assert hd95_brats(b, z, (1, 1, 1)) == HD95_EMPTY_PENALTY


def test_hd95_identical_is_zero_and_shift_is_about_the_shift():
    a = ball((40, 40, 40), (20, 20, 20), 8)
    b = ball((40, 40, 40), (24, 20, 20), 8)
    assert hd95_brats(a, a, (1, 1, 1)) == 0.0
    assert 3.0 <= hd95_brats(a, b, (1, 1, 1)) <= 5.0
    assert hd95_brats(a, b, (2, 2, 2)) > 1.9 * hd95_brats(a, b, (1, 1, 1))    # spacing is used


def test_components():
    m = np.zeros((20, 20, 20), bool)
    m[2:4, 2:4, 2:4] = True
    m[10:12, 10:12, 10:12] = True
    m[4, 4, 4] = True                      # touches the first block diagonally -> same component (26-connectivity)
    assert n_components(m) == 2


def test_label_rule_tc_over_wt_et_over_tc():
    pb = np.zeros((1, 3, 2, 2), bool)
    pb[0, 0] = [[1, 1], [1, 0]]            # WT
    pb[0, 1] = [[0, 1], [1, 0]]            # TC
    pb[0, 2] = [[0, 0], [1, 0]]            # ET
    lab = label_volume(pb)
    assert lab.dtype == np.uint8
    assert lab[0].tolist() == [[2, 1], [3, 0]]


def test_full_grid_uses_the_preprocess_crop():
    crop = compute_crop_indices()
    assert crop == (24, 216, 28, 220)
    lab = np.zeros((155, 192, 192), np.uint8)
    lab[10, 0, 0] = 3                      # first cropped row / column, slice 10
    full = to_full_grid(lab, crop)
    assert full.shape == (240, 240, 155)
    assert full[24, 28, 10] == 3 and full.sum() == 3


def test_nifti_round_trip(tmp_path):
    arr = np.zeros((240, 240, 155), np.uint8)
    arr[100:110, 100:110, 50:60] = 2
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    save_nifti(arr, affine, (1, 1, 1), tmp_path / "x.nii.gz")
    img = nib.load(str(tmp_path / "x.nii.gz"))
    back = np.asanyarray(img.dataobj)
    assert back.shape == (240, 240, 155) and img.get_data_dtype() == np.uint8
    assert (back == arr).all() and np.allclose(img.affine, affine)


def test_resume_only_counts_complete_pairs(tmp_path):
    csv = tmp_path / "r.csv"
    base = dict.fromkeys(COLUMNS, np.nan)
    rows = [dict(base, model="M0", seed=0, patient_id="p1", condition_id="clean"),
            dict(base, model="M4", seed=0, patient_id="p1", condition_id="clean"),
            dict(base, model="M0", seed=0, patient_id="p2", condition_id="clean")]     # p2 misses M4
    append_rows(csv, rows)
    assert completed_pairs(csv, {("M0", 0), ("M4", 0)}) == {("p1", "clean")}
    assert completed_pairs(tmp_path / "missing.csv", {("M0", 0)}) == set()
    assert list(pd.read_csv(csv).columns) == COLUMNS
