"""
Acceptance checks for evaluate_v2 on the VALIDATION patients.
Run on Colab after `python -m evaluation.evaluate_v2 --split val`; needs a GPU only for check 3.

  1. Dice reproduces the training logs: per (model, seed) the mean 3D Dice over the 37 validation
     patients at clean / SNR 5 / worst equals results/train_v3/stepC_summary.csv within 0.005.
  2. HD95 convention: both masks empty -> 0.0, exactly one empty -> 373.13, in the written CSV.
  3. A written NIfTI reloads with the right shape, dtype, labels, affine, and its voxel counts equal
     the in-memory prediction's (the model is re-run on that one volume).
  4. Timing is printed by evaluate_v2 itself (seconds per (patient, condition)).

Usage: python -m evaluation.check_val --raw_dir <MICCAI_BraTS2020_TrainingData> [--csv ...] [--pred_dir ...]
"""

import argparse
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from evaluation.evaluate_v2 import HD95_CONDS, HD95_EMPTY_PENALTY, REGIONS

SUMMARY_COLS = dict(zip(HD95_CONDS, ["clean", "snr5", "worst"]))       # stepC_summary.csv column per condition


def check_dice(csv: Path, summary: Path, tol: float = 0.005) -> bool:
    df, ref = pd.read_csv(csv), pd.read_csv(summary)
    ref["seed"] = ref["seed"].str.replace("seed_", "").astype(int)
    got = (df[df["condition_id"].isin(HD95_CONDS)].groupby(["model", "seed", "condition_id"])["dice_mean"]
           .mean().unstack("condition_id").rename(columns=SUMMARY_COLS).reset_index())
    m = got.merge(ref[["model", "seed", *SUMMARY_COLS.values()]], on=["model", "seed"], suffixes=("", "_log"))
    ok = True
    print("\n[1] Dice vs training logs (best-epoch validation values), tolerance", tol)
    for _, r in m.iterrows():
        diffs = {c: r[c] - r[f"{c}_log"] for c in SUMMARY_COLS.values()}
        good = all(abs(v) <= tol for v in diffs.values())
        ok &= good
        print(f"  {r['model']} seed {int(r['seed'])}: " + "  ".join(
            f"{c} {r[c]:.4f} (log {r[f'{c}_log']:.4f}, diff {diffs[c]:+.4f})" for c in diffs)
            + ("" if good else "   <-- OUT OF TOLERANCE"))
    if len(m) == 0:
        print("  no (model, seed) rows matched the summary")
        return False
    print("  PASS" if ok else "  FAIL")
    return ok


def check_hd95(csv: Path) -> bool:
    df = pd.read_csv(csv)
    df = df[df["condition_id"].isin(HD95_CONDS)]
    ok, n_both, n_one = True, 0, 0
    print("\n[2] HD95 empty-mask convention")
    for r in REGIONS:
        pred, gt, h = df[f"vol_pred_{r}_cm3"], df[f"vol_gt_{r}_cm3"], df[f"hd95_{r}"]
        both = (pred == 0) & (gt == 0)
        one = (pred == 0) ^ (gt == 0)
        n_both, n_one = n_both + int(both.sum()), n_one + int(one.sum())
        ok &= bool((h[both] == 0.0).all()) and bool(np.allclose(h[one], HD95_EMPTY_PENALTY))
        ok &= bool(h.notna().all())
    lgg_no_et = df[(df["vol_gt_et_cm3"] == 0) & (df["grade"] == "LGG")]["patient_id"].nunique()
    print(f"  rows with both masks empty: {n_both} (all 0.0?), exactly one empty: {n_one} (all {HD95_EMPTY_PENALTY}?); "
          f"LGG patients without ET in this set: {lgg_no_et}")
    if n_both == 0:
        print("  NOTE: no both-empty case in this data; the convention is then covered by tests/test_evaluate_v2.py only")
    print("  PASS" if ok else "  FAIL")
    return ok


def check_nifti(csv: Path, pred_dir: Path, raw_dir: str, ckpt_root: str,
                model: str = "M4", cond_id: str = HD95_CONDS[-1], pid: Optional[str] = None) -> bool:
    """Re-run one (patient, condition) in memory and compare with the NIfTI that evaluate_v2 wrote."""
    import torch
    from evaluation.evaluate_v2 import degraded_volume, label_volume, load_models, load_patient, predict, to_full_grid
    from training.train_v2 import DEC005

    df = pd.read_csv(csv)
    pid = pid or sorted(df[df["vol_gt_et_cm3"] > 0]["patient_id"].unique())[0]
    grade = df[df["patient_id"] == pid]["grade"].iloc[0]
    cond = next(c for c in DEC005 if c["id"] == cond_id)
    path = pred_dir / f"{model}_{pid}_{cond_id}.nii.gz"
    print(f"\n[3] NIfTI round trip: {path.name}")
    img = nib.load(str(path))
    lab = np.asanyarray(img.dataobj)
    seg_files = list((Path(raw_dir) / pid).glob("*seg*.nii*"))
    raw_seg = nib.load(str(seg_files[0])) if seg_files else None               # patient 355's is misnamed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_models(ckpt_root, [model], [0], device)[(model, 0)]
    P = load_patient(raw_dir, pid, grade, device)
    pb = predict(net, degraded_volume(P, cond)).cpu().numpy()                   # (Z, 3, H, W)
    full = to_full_grid(label_volume(pb), P["d"]["crop_coords"], P["d"]["orig_shape"])

    checks = {
        "shape is (240, 240, 155)": lab.shape == (240, 240, 155),
        "dtype is uint8": img.get_data_dtype() == np.uint8,
        "labels are within {0, 1, 2, 3}": set(np.unique(lab)) <= {0, 1, 2, 3},
        "file equals the in-memory labels placed on the full grid": bool((lab == full).all()),
        "affine equals the original scan's": raw_seg is None or np.allclose(img.affine, raw_seg.affine),
        "label 3 voxels = predicted ET voxels": int((lab == 3).sum()) == int(pb[:, 2].sum()),
        "voxels > 0 = union of predicted WT / TC / ET": int((lab > 0).sum()) == int(pb.any(axis=1).sum()),
    }
    wt_mem = int(pb[:, 0].sum())
    csv_wt = df[(df.model == model) & (df.seed == 0) & (df.patient_id == pid) & (df.condition_id == cond_id)]
    checks["WT voxels in the CSV volume = in-memory WT voxels"] = bool(np.isclose(
        csv_wt["vol_pred_wt_cm3"].iloc[0] * 1000 / P["vox_cm3"] / 1000, wt_mem, atol=0.5))
    outside = int((lab > 0).sum()) - wt_mem
    print(f"  patient {pid}, {model} seed 0: in-memory WT {wt_mem} voxels, labelled voxels {(lab > 0).sum()} "
          f"({outside} TC/ET voxels lie outside the predicted WT channel; they are labelled 1/3 in the NIfTI)")
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("  PASS" if ok else "  FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--csv", default="results/val_check/raw_metrics_v2_val.csv")
    ap.add_argument("--pred_dir", default=None, help="default: <csv folder>/predictions")
    ap.add_argument("--summary", default="results/train_v3/stepC_summary.csv")
    ap.add_argument("--ckpt_root", default="results/train_v3")
    ap.add_argument("--skip_nifti", action="store_true")
    args = ap.parse_args()
    csv = Path(args.csv)
    pred_dir = Path(args.pred_dir) if args.pred_dir else csv.parent / "predictions"
    results = [check_dice(csv, Path(args.summary)), check_hd95(csv)]
    if not args.skip_nifti:
        results.append(check_nifti(csv, pred_dir, args.raw_dir, args.ckpt_root))
    print("\nALL ACCEPTANCE CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED - do not start the test run")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
