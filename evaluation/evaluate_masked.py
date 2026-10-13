"""
Brain-mask sensitivity analysis (post hoc, not pre-registered): rerun inference on the
74 test patients and score every prediction twice, raw and with everything outside the
brain mask set to zero, for all seeds.

Why: the seed-0 NIfTIs showed that much of M0's false tumor under noise lies outside the
brain (evaluation/check_outside_brain.py, paper 4.4). A clinical pipeline would mask it
away, so the secondary M4 vs M0 comparison is repeated on masked predictions.

Reuses evaluation/evaluate_v2.py unchanged: same preprocessing, same per-(patient,
condition) seeded degradation, same checkpoints, same 0.5 threshold. The brain mask is
the preprocessing mask that also defines the z-score normalisation (any modality > 0).
Raw scores must reproduce results/raw_metrics_v2.csv (checked by --summary).

    python evaluation/evaluate_masked.py --raw_dir <MICCAI_BraTS2020_TrainingData>   # resumable
    python evaluation/evaluate_masked.py --summary                                   # tests + tables

Outputs: results/raw_metrics_masked.csv, tables/masked_secondary_tests.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_v2 import (load_models, load_patient, degraded_volume, predict,  # noqa: E402
                                    prediction_metrics)
from training.train_v2 import VAL_CONDS                                                    # noqa: E402

OUT_CSV = Path("results/raw_metrics_masked.csv")
RAW_CSV = Path("results/raw_metrics_v2.csv")
TABLE = Path("tables/masked_secondary_tests.csv")
KEEP = ["dice_wt", "dice_tc", "dice_et", "dice_mean", "vol_pred_wt_cm3", "false_tumor_cm3"]


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args.ckpt_root, args.models.split(","), [0, 1, 2], device)
    ids = json.load(open(args.splits))["test"][: args.max_patients or None]
    done = set(pd.read_csv(OUT_CSV)["patient_id"]) if OUT_CSV.exists() else set()
    print(f"{device}, {len(models)} checkpoints, {len(ids)} patients ({len(done)} done)", flush=True)
    t_start = time.time()
    for i, pid in enumerate(ids, 1):
        if pid in done:
            continue
        P = load_patient(args.raw_dir, pid, "", device)
        rows = []
        for c in VAL_CONDS:                                   # clean, SNR 5 r 1.0 1 mm, worst
            xn = degraded_volume(P, c)
            for (m, s), net in sorted(models.items()):
                pb = predict(net, xn)
                raw = prediction_metrics(pb, P["gts"], P["vox_cm3"], P["spacing"], with_hd95=False)
                msk = prediction_metrics(pb & P["brain"][:, None], P["gts"], P["vox_cm3"], P["spacing"],
                                         with_hd95=False)
                rows.append(dict(model=m, seed=s, patient_id=pid, condition_id=c["id"],
                                 **{f"{k}_raw": raw[k] for k in KEEP}, **{f"{k}_masked": msk[k] for k in KEEP}))
        pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
        el = time.time() - t_start
        print(f"[{i}/{len(ids)}] {pid}  {el / 60:.1f} min elapsed", flush=True)


def summary():
    from python.analysis.stats import bootstrap_ci, paired_wilcoxon, NOISE_FLOOR_MULTIPLE
    from statsmodels.stats.multitest import multipletests

    d = pd.read_csv(OUT_CSV)
    # 1. Raw scores must reproduce the pre-registered evaluation
    ref = pd.read_csv(RAW_CSV)[["model", "seed", "patient_id", "condition_id", "dice_mean"]]
    j = d.merge(ref, on=["model", "seed", "patient_id", "condition_id"])
    diff = (j["dice_mean_raw"] - j["dice_mean"]).abs()
    print(f"Raw vs raw_metrics_v2 (n={len(j)}): max |dDice| {diff.max():.4f}, "
          f"{(diff > 0.001).mean() * 100:.1f} % of rows differ by > 0.001")

    # 2. Mean Dice per model and condition, raw vs masked (seed-averaged per patient)
    pl = d.groupby(["model", "condition_id", "patient_id"]).mean(numeric_only=True).reset_index()
    print(pl.groupby(["model", "condition_id"])[["dice_mean_raw", "dice_mean_masked", "dice_wt_raw",
                                                  "dice_wt_masked", "false_tumor_cm3_raw",
                                                  "false_tumor_cm3_masked"]].mean().round(3).to_string())

    # 3. M4 vs M0 on masked Dice, same procedure as the DEC-006 secondary tests
    rows = []
    for c in [v["id"] for v in VAL_CONDS]:
        a = pl[(pl.model == "M4") & (pl.condition_id == c)].set_index("patient_id")["dice_mean_masked"]
        b = pl[(pl.model == "M0") & (pl.condition_id == c)].set_index("patient_id")["dice_mean_masked"]
        x = (a - b.reindex(a.index)).to_numpy()
        mean, lo, hi = bootstrap_ci(x, stat=np.mean)
        stat, p = paired_wilcoxon(x)
        sd = max(d[(d.model == m) & (d.condition_id == c)].groupby("seed")["dice_mean_masked"].mean().std(ddof=1)
                 for m in ["M4", "M0"])
        rows.append(dict(test_model="M4", ref_model="M0", condition=c, metric="dice_mean_masked",
                         n_patients=len(x), mean_diff=mean, ci95_low=lo, ci95_high=hi,
                         median_diff=float(np.median(x)), wilcoxon_stat=stat, raw_p=p, seed_noise_floor=sd,
                         within_training_noise=bool(abs(mean) < NOISE_FLOOR_MULTIPLE * sd)))
    t = pd.DataFrame(rows)
    t["p_holm"] = multipletests(t["raw_p"], method="holm")[1]
    TABLE.parent.mkdir(exist_ok=True)
    t.to_csv(TABLE, index=False)
    print(t[["condition", "mean_diff", "ci95_low", "ci95_high", "raw_p", "p_holm", "seed_noise_floor",
             "within_training_noise"]].round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--ckpt_root", default="results/train_v3")
    ap.add_argument("--splits", default="data/splits/split_v1.json")
    ap.add_argument("--models", default="M0,M1,M4,M5,M6")
    ap.add_argument("--max_patients", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    summary() if a.summary else run(a)
