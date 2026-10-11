"""
Step D: evaluate the 15 trained models (M0, M1, M4, M5, M6 x seeds 0, 1, 2) on full 3D volumes
under the 13 DEC-005 conditions (docs/DECISIONS.md DEC-005, DEC-007; brief: docs/handoff/TASK_NAZIR_STEP_D.md).

Inference recipe is exactly training.train_v2.validate(): degrade_volume -> normalise -> model in
slices of 32 -> sigmoid output > 0.5 -> WT / TC / ET. Each (patient, condition) volume is degraded
and normalised ONCE and every checkpoint runs on that same volume; the noise is seeded by
pilot.pilot.seed_for(patient, condition), as in training validation, so all comparisons are paired.

Output, one row per (model, seed, patient, condition), appended to --out_csv:
  model, seed, patient_id, grade, condition_id, noise, snr, r, thick,
  dice_wt, dice_tc, dice_et, dice_mean,
  vol_pred_{wt,tc,et}_cm3, vol_gt_{wt,tc,et}_cm3     (voxel volume from the NIfTI header spacing)
  n_components_wt                                    (26-connected components of the predicted WT)
  false_tumor_cm3                                    (predicted WT on slices with no true tumor)
  hd95_{wt,tc,et}                                    (only for HD95_CONDS; empty elsewhere)
For clean, snr is written as 0 (as in the pilot CSV); condition_id is the key.

HD95 uses the BraTS convention locally (finding C9): both masks empty -> 0.0, exactly one empty -> 373.13.

Seed-0 predictions at HD95_CONDS are also written as NIfTI for the MATLAB pipeline:
  <out_dir>/predictions/<model>_<patient_id>_<condition_id>.nii.gz        uint8, 240 x 240 x 155
  <out_dir>/predictions/ground_truth/GT_<patient_id>.nii.gz
Labels 0 / 1 = NCR / 2 = edema / 3 = ET, from WT -> 2, then TC -> 1, then ET -> 3. The 192 x 192
crop is placed back with the crop coordinates of python.data.preprocess.compute_crop_indices
(rows 24:216, columns 28:220) and the original affine.

Resumable: a (patient, condition) pair is skipped when the CSV already holds a row for every
loaded (model, seed). NIfTI files are written before the CSV rows, so a crash never leaves rows
without predictions.

Validation is the default. The test set needs --confirm_test and all 15 checkpoints: it is run once.

Usage (from the repo root):
  python -m evaluation.evaluate_v2 --raw_dir <MICCAI_BraTS2020_TrainingData> --split val
  python -m evaluation.evaluate_v2 --raw_dir <...> --split test --confirm_test
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from medpy.metric.binary import hd95 as _medpy_hd95
from scipy import ndimage

from pilot.pilot import degrade_volume, dice, normalise, seed_for, self_check
from python.data.preprocess import compute_crop_indices, preprocess_patient
from python.models.factory import build_model
from training.prepare import patient_row
from training.train_v2 import DEC005, VAL_CONDS

MODELS = ["M0", "M1", "M4", "M5", "M6"]
SEEDS = [0, 1, 2]
HD95_CONDS = [c["id"] for c in VAL_CONDS]          # clean, SNR 5 r 1.0 1 mm, worst (SNR 3 r 0.5 5 mm)
NIFTI_CONDS = HD95_CONDS
HD95_EMPTY_PENALTY = 373.13                          # BraTS: exactly one of the two masks is empty
REGIONS = ["wt", "tc", "et"]

COLUMNS = (["model", "seed", "patient_id", "grade", "condition_id", "noise", "snr", "r", "thick",
            "dice_wt", "dice_tc", "dice_et", "dice_mean"]
           + [f"vol_{w}_{r}_cm3" for r in REGIONS for w in ("pred", "gt")]
           + ["n_components_wt", "false_tumor_cm3", "hd95_wt", "hd95_tc", "hd95_et"])


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------
def hd95_brats(pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    """95th-percentile Hausdorff distance (mm) with the BraTS empty-mask convention (finding C9).

    pred, gt: boolean arrays ordered like `spacing` (here H, W, Z). Both empty -> 0.0; exactly one
    empty -> 373.13. python.metrics.evaluation.compute_3d_hd95 returns 373.13 for both-empty too,
    which charges a correct "no tumor" prediction the full penalty; that file is left untouched.
    """
    p, g = pred.astype(bool), gt.astype(bool)
    n_p, n_g = p.any(), g.any()
    if not n_p and not n_g:
        return 0.0
    if not n_p or not n_g:
        return HD95_EMPTY_PENALTY
    coords = np.argwhere(p | g)
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    return float(_medpy_hd95(p[box], g[box], voxelspacing=spacing))


def n_components(mask: np.ndarray) -> int:
    """Number of 26-connected components of a 3D boolean mask."""
    return int(ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=int))[1])


def region_gts(seg: torch.Tensor) -> List[torch.Tensor]:
    """seg (Z, H, W) with labels 0/1/2/3 -> boolean WT, TC, ET (same definition as training)."""
    return [seg > 0, (seg == 1) | (seg == 3), seg == 3]


def prediction_metrics(pb: torch.Tensor, gts: List[torch.Tensor], vox_cm3: float,
                       spacing: Tuple[float, float, float], with_hd95: bool) -> Dict[str, float]:
    """All per-prediction metrics. pb: (Z, 3, H, W) bool, gts: three (Z, H, W) bool, same device."""
    out: Dict[str, float] = {}
    ds = [dice(pb[:, k], gts[k]) for k in range(3)]
    out.update(dice_wt=ds[0], dice_tc=ds[1], dice_et=ds[2], dice_mean=float(np.mean(ds)))
    for k, r in enumerate(REGIONS):
        out[f"vol_pred_{r}_cm3"] = pb[:, k].sum().item() * vox_cm3
        out[f"vol_gt_{r}_cm3"] = gts[k].sum().item() * vox_cm3
    tumor_free = gts[0].sum(dim=(1, 2)) == 0
    out["false_tumor_cm3"] = pb[tumor_free, 0].sum().item() * vox_cm3
    out["n_components_wt"] = n_components(pb[:, 0].cpu().numpy())
    if with_hd95:
        for k, r in enumerate(REGIONS):
            p = np.ascontiguousarray(pb[:, k].permute(1, 2, 0).cpu().numpy())    # (H, W, Z)
            g = np.ascontiguousarray(gts[k].permute(1, 2, 0).cpu().numpy())
            out[f"hd95_{r}"] = hd95_brats(p, g, spacing)
    return out


# ---------------------------------------------------------------------------------------------
# NIfTI export
# ---------------------------------------------------------------------------------------------
def label_volume(pb: np.ndarray) -> np.ndarray:
    """(Z, 3, H, W) bool WT/TC/ET -> (Z, H, W) uint8 labels. WT -> 2, TC over WT -> 1, ET over TC -> 3."""
    lab = np.zeros(pb.shape[:1] + pb.shape[2:], dtype=np.uint8)
    lab[pb[:, 0]] = 2
    lab[pb[:, 1]] = 1
    lab[pb[:, 2]] = 3
    return lab


def to_full_grid(lab_zhw: np.ndarray, crop: Tuple[int, int, int, int],
                 orig_shape: Tuple[int, int, int] = (240, 240, 155)) -> np.ndarray:
    """Place the cropped (Z, H, W) labels back onto the (X, Y, Z) grid of the original scan.

    Uses the crop of python.data.preprocess (rows 24:216, columns 28:220). Do NOT use the default
    crop_coords of python.data.dataset.reconstruct_3d_volume: its columns are 24:216 (4 voxels off).
    """
    r_s, r_e, c_s, c_e = crop
    full = np.zeros(orig_shape, dtype=np.uint8)
    full[r_s:r_e, c_s:c_e, :] = lab_zhw.transpose(1, 2, 0)
    return full


def save_nifti(arr: np.ndarray, affine: np.ndarray, zooms, path: Path) -> None:
    img = nib.Nifti1Image(arr.astype(np.uint8), affine)
    img.header.set_data_dtype(np.uint8)
    img.header.set_zooms(tuple(float(z) for z in zooms[:3]))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def save_ground_truth(seg_path: str, path: Path) -> None:
    """Full-grid ground truth with the same label scheme as the predictions (BraTS 4 -> 3)."""
    img = nib.load(seg_path)
    seg = np.asanyarray(img.dataobj, dtype=np.uint8)
    out = np.zeros_like(seg)
    out[seg == 1], out[seg == 2], out[seg == 4] = 1, 2, 3
    save_nifti(out, img.affine, img.header.get_zooms(), path)


# ---------------------------------------------------------------------------------------------
# models and inference
# ---------------------------------------------------------------------------------------------
def load_models(ckpt_root: str, models: List[str], seeds: List[int], device) -> Dict[Tuple[str, int], torch.nn.Module]:
    """Every <ckpt_root>/<model>/seed_<k>/best_model.pt that exists, all on the device at once."""
    loaded = {}
    for m in models:
        for s in seeds:
            f = Path(ckpt_root) / m / f"seed_{s}" / "best_model.pt"
            if not f.exists():
                continue
            net = build_model(m, d0=0.20).to(device)
            # weights_only=False: our own checkpoints hold more than tensors (PyTorch >= 2.6 default refuses)
            net.load_state_dict(torch.load(f, map_location=device, weights_only=False)["model_state_dict"])
            loaded[(m, s)] = net.eval()
    return loaded


@torch.no_grad()
def predict(net: torch.nn.Module, xn: torch.Tensor, batch: int = 32) -> torch.Tensor:
    """(Z, C, H, W) normalised volume -> (Z, 3, H, W) bool. Same as train_v2.validate()."""
    with torch.autocast(xn.device.type, enabled=xn.device.type == "cuda"):
        p = torch.cat([net(xn[s:s + batch]) for s in range(0, xn.shape[0], batch)])
    return p.float() > 0.5


# ---------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------
def load_patient(raw_dir: str, pid: str, grade: str, device) -> Dict:
    """Preprocess one patient (same crop / mask / mu as training) and put it on the device."""
    row = patient_row(Path(raw_dir), pid)
    row["grade"] = grade
    d = preprocess_patient(row)
    spacing = tuple(float(z) for z in d["zooms"])                               # (dx, dy, dz) mm
    seg = torch.from_numpy(d["seg"]).permute(2, 0, 1).to(device)                # (Z, H, W)
    return dict(pid=pid, row=row, d=d, device=device, spacing=spacing, vox_cm3=float(np.prod(spacing)) / 1000.0,
                x=torch.from_numpy(d["modalities"].astype(np.float32)).permute(3, 0, 1, 2).contiguous().to(device),
                brain=torch.from_numpy(d["brain_mask"]).permute(2, 0, 1).bool().to(device),
                mu=torch.from_numpy(d["mu_brain"]).to(device), seg=seg, gts=region_gts(seg))


@torch.no_grad()
def degraded_volume(P: Dict, cond: Dict) -> torch.Tensor:
    """Degrade + normalise one patient under one condition, seeded per (patient, condition)."""
    gen = torch.Generator(device=P["device"]).manual_seed(seed_for(P["pid"], cond["id"]))
    return normalise(degrade_volume(P["x"], P["mu"], cond, gen), P["brain"])


def completed_pairs(csv_path: Path, keys: set) -> set:
    """(patient, condition) pairs that already have a row for every (model, seed) in `keys`."""
    if not csv_path.exists():
        return set()
    prev = pd.read_csv(csv_path, usecols=["model", "seed", "patient_id", "condition_id"])
    prev["key"] = list(zip(prev["model"], prev["seed"].astype(int)))
    have = prev.groupby(["patient_id", "condition_id"])["key"].agg(set)
    return {pc for pc, ks in have.items() if keys <= ks}


def append_rows(csv_path: Path, rows: List[Dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def run(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    self_check(device)

    models = load_models(args.ckpt_root, args.models.split(","), [int(s) for s in args.seeds.split(",")], device)
    if not models:
        raise SystemExit(f"No checkpoints found under {args.ckpt_root}/<model>/seed_<k>/best_model.pt")
    keys = set(models)
    print(f"Loaded {len(models)} checkpoints: " + ", ".join(f"{m}/s{s}" for m, s in sorted(models)), flush=True)
    if args.split == "test":
        missing = [(m, s) for m in MODELS for s in SEEDS if (m, s) not in keys]
        if missing:
            raise SystemExit(f"Test set is run once, with all 15 checkpoints. Missing: {missing}")

    ids = json.load(open(args.splits))[args.split]
    if args.max_patients:
        ids = ids[: args.max_patients]
    conds = [c for c in DEC005 if not args.conditions or c["id"] in args.conditions.split(",")]
    mapping = pd.read_csv(Path(args.raw_dir) / "name_mapping.csv")
    grade = dict(zip(mapping["BraTS_2020_subject_ID"], mapping["Grade"]))

    out_csv = Path(args.out_csv)
    pred_dir = Path(args.out_dir) / "predictions"
    done = completed_pairs(out_csv, keys)
    print(f"{args.split}: {len(ids)} patients x {len(conds)} conditions x {len(models)} checkpoints; "
          f"{len(done)} (patient, condition) pairs already in {out_csv}", flush=True)

    timings, t_start = [], time.time()
    for i, pid in enumerate(ids, 1):
        todo = [c for c in conds if (pid, c["id"]) not in done]
        if not todo:
            continue
        if args.time_limit_h and (time.time() - t_start) / 3600 > args.time_limit_h:
            print(f"Stopping at {(time.time() - t_start) / 3600:.1f} h (time limit): run again with this output "
                  f"as input to continue; finished (patient, condition) pairs are skipped.", flush=True)
            break
        P = load_patient(args.raw_dir, pid, grade.get(pid, "Unknown"), device)
        row, d, x, brain, mu, gts = P["row"], P["d"], P["x"], P["brain"], P["mu"], P["gts"]
        spacing, vox_cm3 = P["spacing"], P["vox_cm3"]
        if not args.no_nifti and any(c["id"] in NIFTI_CONDS for c in todo):
            save_ground_truth(row["seg_path"], pred_dir / "ground_truth" / f"GT_{pid}.nii.gz")

        for c in todo:
            t0 = time.time()
            xn = degraded_volume(P, c)
            rows = []
            for (m, s), net in sorted(models.items()):
                pb = predict(net, xn, args.batch)
                met = prediction_metrics(pb, gts, vox_cm3, spacing, with_hd95=c["id"] in HD95_CONDS)
                rows.append(dict(model=m, seed=s, patient_id=pid, grade=row["grade"], condition_id=c["id"],
                                 noise=c["noise"], snr=c["snr"] or 0, r=c["r"], thick=c["thick"], **met))
                if s == 0 and c["id"] in NIFTI_CONDS and not args.no_nifti:
                    lab = label_volume(pb.cpu().numpy())
                    save_nifti(to_full_grid(lab, d["crop_coords"], d["orig_shape"]), d["affine"], d["zooms"],
                               pred_dir / f"{m}_{pid}_{c['id']}.nii.gz")
            append_rows(out_csv, rows)                                           # after the NIfTI files
            timings.append(time.time() - t0)
        print(f"[{i}/{len(ids)}] {pid} ({row['grade']}): {len(todo)} conditions, "
              f"{np.mean(timings[-len(todo):]):.1f} s per (patient, condition), "
              f"{(time.time() - t_start) / 60:.1f} min elapsed", flush=True)

    if timings:
        print(f"\nTIMING: {np.mean(timings):.1f} s per (patient, condition) for {len(models)} models "
              f"(median {np.median(timings):.1f} s). Full test run: 74 x 13 = 962 pairs -> "
              f"{962 * np.mean(timings) / 3600:.1f} h at this speed.", flush=True)
    finished = completed_pairs(out_csv, keys)
    left = sum(1 for pid in ids for c in conds if (pid, c["id"]) not in finished)
    print("EVALUATION DONE" if left == 0 else f"EVALUATION INCOMPLETE: {left} (patient, condition) pairs left", flush=True)


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", required=True, help="MICCAI_BraTS2020_TrainingData folder (with name_mapping.csv)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm_test", action="store_true", help="required for --split test (run it once)")
    ap.add_argument("--ckpt_root", default="results/train_v3")
    ap.add_argument("--splits", default="data/splits/split_v1.json")
    ap.add_argument("--out_csv", default=None, help="default: results/raw_metrics_v2.csv (test) or "
                    "results/val_check/raw_metrics_v2_val.csv (val)")
    ap.add_argument("--out_dir", default=None, help="parent of predictions/ (default: folder of --out_csv)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--conditions", default="", help="comma-separated DEC-005 ids (default: all 13)")
    ap.add_argument("--max_patients", type=int, default=0, help="0 = all patients of the split")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--time_limit_h", type=float, default=0, help="do not START a new patient after this many hours "
                    "(Kaggle stops a run at 12 h); 0 = no limit")
    ap.add_argument("--no_nifti", action="store_true", help="skip the NIfTI export (debugging)")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("The test set is evaluated exactly once. Re-run with --confirm_test when all "
                         "15 checkpoints exist and validation checks have passed.")
    if args.out_csv is None:
        args.out_csv = ("results/raw_metrics_v2.csv" if args.split == "test"
                        else "results/val_check/raw_metrics_v2_val.csv")
    if args.out_dir is None:
        args.out_dir = str(Path(args.out_csv).parent)
    run(args)


if __name__ == "__main__":
    main()
