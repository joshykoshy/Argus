"""
Prepare BraTS patients for fast slice-level training (run once per Colab session).

For each patient in the split (train + val by default), runs the team's
python.data.preprocess.preprocess_patient (same crop, label remap 4 -> 3, brain
mask, mu_brain) and stores slice-major arrays that can be memory-mapped, so a
training batch reads single slices without loading whole volumes:
  <work>/<id>_x.npy   float16 (155, 4, 192, 192)   raw intensities
  <work>/<id>_y.npy   uint8   (155, 192, 192)      labels 0/1/2/3
  <work>/<id>_m.npy   bool    (155, 192, 192)      brain mask
  <work>/<id>_mu.npy  float32 (4,)                 mean brain intensity
Existing patients are skipped, so an interrupted run resumes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from python.data.preprocess import preprocess_patient


def patient_row(raw_dir: Path, pid: str) -> pd.Series:
    f = {p.name.split("_")[-1].split(".")[0]: str(p) for p in (raw_dir / pid).glob("*.nii*")}
    # Patient 355's seg is misnamed in the Kaggle copy (W39_1998.09.19_Segm.nii).
    seg = f.get("seg") or next(str(p) for p in (raw_dir / pid).glob("*.nii*")
                               if not any(k in p.name for k in ["_t1", "_t2", "_flair"]))
    return pd.Series(dict(patient_id=pid, grade="", t1_path=f["t1"], t1ce_path=f["t1ce"],
                          t2_path=f["t2"], flair_path=f["flair"], seg_path=seg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--splits", default="data/splits/split_v1.json")
    ap.add_argument("--sets", default="train,val")
    args = ap.parse_args()
    raw, work = Path(args.raw_dir), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    split = json.load(open(args.splits))
    ids = [p for s in args.sets.split(",") for p in split[s]]
    for i, pid in enumerate(ids, 1):
        if (work / f"{pid}_mu.npy").exists():
            continue
        d = preprocess_patient(patient_row(raw, pid))
        np.save(work / f"{pid}_x.npy", np.ascontiguousarray(d["modalities"].transpose(3, 0, 1, 2)))
        np.save(work / f"{pid}_y.npy", np.ascontiguousarray(d["seg"].transpose(2, 0, 1)))
        np.save(work / f"{pid}_m.npy", np.ascontiguousarray(d["brain_mask"].transpose(2, 0, 1).astype(bool)))
        np.save(work / f"{pid}_mu.npy", d["mu_brain"])   # written last = completion marker
        if i % 20 == 0 or i == len(ids):
            print(f"prepared {i}/{len(ids)}", flush=True)
    print("PREPARE DONE")


if __name__ == "__main__":
    main()
