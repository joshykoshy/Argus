"""
Statistics for the ECTE408 study, to the pre-registered DEC-006 (docs/DECISIONS.md).

Input: results/raw_metrics_v2.csv from evaluation/evaluate_v2.py (74 test patients, 13 DEC-005
conditions, models M0 M1 M4 M5 M6, seeds 0-2). Every test uses full-volume 3D Dice (mean of WT, TC, ET),
averaged over the three seeds per patient, and a paired two-sided Wilcoxon signed-rank test.

  H1 (primary)  M4 > M1 at the worst condition (SNR 3, r 0.5, 5 mm, k-space noise)
  H2            M4's clean-to-worst Dice drop is smaller than M1's (per-patient drops)
  H3            M4 > M6 at the worst condition (two streams vs early fusion)

Holm correction over these three, alpha 0.05. A hypothesis counts as SUPPORTED only if Holm p < alpha,
the effect is in the predicted direction, and it is NOT within training noise: a difference smaller than
twice the seed-to-seed SD is reported as "within training noise" regardless of p (DEC-006).
Direction: tests are two-sided and the sign of the mean paired difference must match the hypothesis, which is
at least as strict as a one-sided test.

Secondary (reported, outside the verdict): M4 vs M0 and M5 vs M4 at clean and worst; the 13-condition
Dice curves of every model with 95 % bootstrap CIs over patients (seed 42); headline table with HD95
median / IQR (HD95 exists only for clean, SNR 5 r 1.0 1 mm, worst; both-empty = 0 and one-empty = 373.13 mm).

Usage: python -m python.analysis.stats [raw_metrics_v2.csv] [tables_dir]
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RAW_METRICS_CSV = "results/raw_metrics_v2.csv"
TABLES_DIR = "tables"

CLEAN = "clean"
MID = "kspace_snr5_r1.0_t1"
WORST = "kspace_snr3_r0.5_t5"
ALPHA = 0.05
NOISE_FLOOR_MULTIPLE = 2.0
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
METRIC = "dice_mean"

# DEC-006. kind "level": test - ref at `condition`; kind "drop": (clean - worst) of ref minus that of test,
# so a positive effect always means "test is better" and every hypothesis predicts effect > 0.
HYPOTHESES = {
    "H1": dict(test="M4", ref="M1", kind="level", condition=WORST,
               text="M4 beats M1 at the worst condition (SNR 3, r 0.5, 5 mm)"),
    "H2": dict(test="M4", ref="M1", kind="drop", condition=WORST,
               text="M4's clean-to-worst Dice drop is smaller than M1's"),
    "H3": dict(test="M4", ref="M6", kind="level", condition=WORST,
               text="M4 beats M6 at the worst condition (two streams vs early fusion)"),
}
# Secondary: reported, not part of the verdict.
SECONDARY = [("M4", "M0", WORST), ("M4", "M0", CLEAN), ("M5", "M4", WORST), ("M5", "M4", CLEAN)]
HEADLINE_CONDITIONS = [CLEAN, MID, WORST]

DICE_COLS = ["dice_wt", "dice_tc", "dice_et", "dice_mean"]
HD_COLS = ["hd95_wt", "hd95_tc", "hd95_et"]
REQUIRED = ["model", "seed", "patient_id", "grade", "condition_id", "noise", "snr", "r", "thick"] + DICE_COLS


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(x: np.ndarray, stat=np.median, n: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED, ci: float = 0.95) -> Tuple[float, float, float]:
    """(point estimate, lo, hi) of `stat` over patients, percentile bootstrap."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boot = stat(x[idx], axis=1)
    a = (1 - ci) / 2
    return float(stat(x)), float(np.percentile(boot, 100 * a)), float(np.percentile(boot, 100 * (1 - a)))


def paired_wilcoxon(diffs: np.ndarray) -> Tuple[float, float]:
    """(statistic, two-sided p); p = 1 when all differences are zero."""
    d = np.asarray(diffs, float)
    if len(d) == 0 or np.all(d == 0):
        return 0.0, 1.0
    res = wilcoxon(d, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def patient_level(df: pd.DataFrame) -> pd.DataFrame:
    """Average the seeds per (model, patient, condition); HD95 stays NaN where it was not computed."""
    cols = [c for c in DICE_COLS + HD_COLS + ["false_tumor_cm3"] if c in df.columns]
    return df.groupby(["model", "patient_id", "grade", "condition_id"], as_index=False)[cols].mean()


def _series(pl: pd.DataFrame, model: str, cond: str, metric: str = METRIC) -> pd.Series:
    sub = pl[(pl["model"] == model) & (pl["condition_id"] == cond)]
    return sub.set_index("patient_id")[metric].sort_index()


def paired_level_diff(pl: pd.DataFrame, test: str, ref: str, cond: str, metric: str = METRIC) -> pd.Series:
    """test - ref per patient, aligned on patient_id (never by position)."""
    a, b = _series(pl, test, cond, metric), _series(pl, ref, cond, metric)
    common = a.index.intersection(b.index)
    return a.loc[common] - b.loc[common]


def drop_per_patient(pl: pd.DataFrame, model: str, cond: str = WORST, metric: str = METRIC) -> pd.Series:
    """clean - degraded, per patient (positive = performance lost)."""
    c, d = _series(pl, model, CLEAN, metric), _series(pl, model, cond, metric)
    common = c.index.intersection(d.index)
    return c.loc[common] - d.loc[common]


def paired_drop_diff(pl: pd.DataFrame, test: str, ref: str, cond: str = WORST, metric: str = METRIC) -> pd.Series:
    """drop(ref) - drop(test) per patient; positive = `test` loses less."""
    a, b = drop_per_patient(pl, test, cond, metric), drop_per_patient(pl, ref, cond, metric)
    common = a.index.intersection(b.index)
    return b.loc[common] - a.loc[common]


# ---------------------------------------------------------------------------------------------
# seed noise floor
# ---------------------------------------------------------------------------------------------
def seed_noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """SD across seeds of each model's mean Dice (mean over patients per seed), per condition and metric."""
    rows = []
    for (model, cond), g in df.groupby(["model", "condition_id"]):
        for metric in DICE_COLS:
            per_seed = g.groupby("seed")[metric].mean()
            rows.append({"model": model, "condition": cond, "metric": metric, "n_seeds": len(per_seed),
                         "seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan})
    return pd.DataFrame(rows)


def seed_sd_of_drop(df: pd.DataFrame, model: str, cond: str = WORST, metric: str = METRIC) -> float:
    """SD across seeds of a model's clean-to-`cond` drop in mean Dice."""
    g = df[df["model"] == model]
    per_seed = (g[g["condition_id"] == CLEAN].groupby("seed")[metric].mean()
                - g[g["condition_id"] == cond].groupby("seed")[metric].mean())
    return float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan


def _level_floor(floor_df: pd.DataFrame, models: List[str], cond: str, metric: str = METRIC) -> float:
    r = floor_df[(floor_df["model"].isin(models)) & (floor_df["condition"] == cond) & (floor_df["metric"] == metric)]
    return float(r["seed_sd"].max()) if len(r) else np.nan


def _within_noise(effect: float, floor: float) -> bool:
    return bool(abs(effect) < NOISE_FLOOR_MULTIPLE * floor) if not (np.isnan(effect) or np.isnan(floor)) else False


# ---------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------
def _test_record(d: pd.Series, floor: float) -> Dict:
    """Paired summary of one comparison from per-patient differences `d` (positive = test better)."""
    x = d.to_numpy()
    mean, lo, hi = bootstrap_ci(x, stat=np.mean)
    med, med_lo, med_hi = bootstrap_ci(x, stat=np.median)
    stat, p = paired_wilcoxon(x)
    return {"n_patients": len(x), "mean_diff": mean, "ci95_low": lo, "ci95_high": hi, "median_diff": med,
            "median_ci95_low": med_lo, "median_ci95_high": med_hi, "wilcoxon_stat": stat, "raw_p": p,
            "seed_noise_floor": floor, "within_training_noise": _within_noise(mean, floor)}


def _diff_and_floor(pl: pd.DataFrame, df: pd.DataFrame, floor_df: pd.DataFrame, test: str, ref: str,
                    kind: str, cond: str) -> Tuple[pd.Series, float]:
    if kind == "level":
        return (paired_level_diff(pl, test, ref, cond), _level_floor(floor_df, [test, ref], cond))
    floor = np.nanmax([seed_sd_of_drop(df, test, cond), seed_sd_of_drop(df, ref, cond)])
    return paired_drop_diff(pl, test, ref, cond), float(floor)


def evaluate_hypotheses(pl: pd.DataFrame, df: pd.DataFrame, floor_df: pd.DataFrame) -> pd.DataFrame:
    """H1-H3 with Holm over the three raw p-values and the DEC-006 verdict rule."""
    recs = []
    for name, h in HYPOTHESES.items():
        d, floor = _diff_and_floor(pl, df, floor_df, h["test"], h["ref"], h["kind"], h["condition"])
        recs.append({"hypothesis": name, "text": h["text"], "test_model": h["test"], "ref_model": h["ref"],
                     "kind": h["kind"], "condition": h["condition"], **_test_record(d, floor)})
    out = pd.DataFrame(recs)
    out["p_holm"] = holm(out["raw_p"].to_numpy())
    sig = out["p_holm"] < ALPHA
    pos = out["mean_diff"] > 0
    out["verdict"] = np.where(sig & pos & ~out["within_training_noise"], "SUPPORTED", "NOT SUPPORTED")
    out["note"] = np.select(
        [sig & pos & out["within_training_noise"], sig & ~pos],
        ["statistically significant but within training noise (|diff| < 2 x seed SD): not counted",
         "significant in the OPPOSITE direction"], default="")
    return out


def secondary_tests(pl: pd.DataFrame, df: pd.DataFrame, floor_df: pd.DataFrame) -> pd.DataFrame:
    """Reported, not part of the verdict; Holm only within this secondary family."""
    recs = []
    for test, ref, cond in SECONDARY:
        d, floor = _diff_and_floor(pl, df, floor_df, test, ref, "level", cond)
        recs.append({"test_model": test, "ref_model": ref, "condition": cond, **_test_record(d, floor)})
    out = pd.DataFrame(recs)
    out["p_holm_secondary"] = holm(out["raw_p"].to_numpy())
    out["note"] = np.where(out["within_training_noise"], "within training noise", "")
    return out


# ---------------------------------------------------------------------------------------------
# curves and headline table
# ---------------------------------------------------------------------------------------------
def dice_curves(pl: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Mean Dice per model x condition (seed-averaged per patient) with 95 % bootstrap CI over patients."""
    meta = df.drop_duplicates("condition_id").set_index("condition_id")[["noise", "snr", "r", "thick"]]
    rows = []
    for (model, cond), g in pl.groupby(["model", "condition_id"]):
        for metric in DICE_COLS:
            mean, lo, hi = bootstrap_ci(g[metric].to_numpy(), stat=np.mean)
            rows.append({"model": model, "condition": cond, "metric": metric, "n_patients": len(g),
                         "mean": mean, "ci95_low": lo, "ci95_high": hi, **meta.loc[cond].to_dict()})
    return pd.DataFrame(rows)


def headline_table(df: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(df["model"].unique()):
        for cond in HEADLINE_CONDITIONS:
            sub = df[(df["model"] == model) & (df["condition_id"] == cond)]
            psub = pl[(pl["model"] == model) & (pl["condition_id"] == cond)]
            row = {"model": model, "condition": cond, "n_seeds": sub["seed"].nunique(),
                   "n_patients": psub["patient_id"].nunique()}
            for metric in DICE_COLS:
                s = sub.groupby("seed")[metric].mean()
                row[f"{metric}_mean"] = s.mean()
                row[f"{metric}_seed_sd"] = s.std(ddof=1) if len(s) > 1 else np.nan
            for metric in HD_COLS:
                v = psub[metric].dropna() if metric in psub else pd.Series(dtype=float)
                row[f"{metric}_median"] = v.median()
                row[f"{metric}_q25"] = v.quantile(0.25)
                row[f"{metric}_q75"] = v.quantile(0.75)
            if "false_tumor_cm3" in sub:
                row["false_tumor_cm3_mean"] = sub["false_tumor_cm3"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------
def run_statistical_tests(raw_metrics_csv: str = RAW_METRICS_CSV, tables_dir: str = TABLES_DIR) -> Dict:
    df = pd.read_csv(raw_metrics_csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{raw_metrics_csv} is missing columns: {missing}")
    need = {m for h in HYPOTHESES.values() for m in (h["test"], h["ref"])} | {m for t in SECONDARY for m in t[:2]}
    absent = sorted(need - set(df["model"].unique()))
    if absent:
        raise ValueError(f"models missing from the data: {absent}")
    conds = {CLEAN, MID, WORST} - set(df["condition_id"].unique())
    if conds:
        raise ValueError(f"conditions missing from the data: {sorted(conds)}")
    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pl = patient_level(df)
    n_by_model = pl[pl["condition_id"] == WORST].groupby("model")["patient_id"].nunique()
    if n_by_model.nunique() > 1:
        print(f"WARNING: models differ in patient count at the worst condition:\n{n_by_model}")
    floor_df = seed_noise_floor(df)

    hyp = evaluate_hypotheses(pl, df, floor_df)
    sec = secondary_tests(pl, df, floor_df)
    curves = dice_curves(pl, df)
    headline = headline_table(df, pl)

    hyp.to_csv(out_dir / "hypotheses.csv", index=False)
    sec.to_csv(out_dir / "secondary_tests.csv", index=False)
    curves.to_csv(out_dir / "dice_curves.csv", index=False)
    headline.to_csv(out_dir / "headline_metrics.csv", index=False)
    floor_df.to_csv(out_dir / "seed_noise_floor.csv", index=False)

    print("\n================ HYPOTHESIS VERDICTS (DEC-006) ================")
    for _, r in hyp.iterrows():
        print(f"{r['hypothesis']}: {r['verdict']} - {r['text']}\n    mean diff {r['mean_diff']:+.4f} "
              f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}], Holm p = {r['p_holm']:.3g} (raw {r['raw_p']:.3g}), "
              f"2 x seed SD = {NOISE_FLOOR_MULTIPLE * r['seed_noise_floor']:.4f}"
              + (f"\n    NOTE: {r['note']}" if r["note"] else ""))
    print("===============================================================\n")
    return {"hypotheses": hyp, "secondary_tests": sec, "dice_curves": curves, "headline_metrics": headline,
            "seed_noise_floor": floor_df}


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else RAW_METRICS_CSV
    tdir = sys.argv[2] if len(sys.argv) > 2 else TABLES_DIR
    run_statistical_tests(csv, tdir)
