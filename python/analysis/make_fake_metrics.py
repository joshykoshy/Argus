"""
Synthetic raw_metrics_v2.csv with KNOWN planted effects, for testing python/analysis/stats.py (DEC-006).

Same columns as evaluation/evaluate_v2.py: 74 patients x 13 DEC-005 conditions x models M0 M1 M4 M5 M6
x seeds 0-2. HD95 is filled only at clean, SNR 5 r 1.0 1 mm and the worst condition (empty elsewhere),
and is 0.0 for the five LGG patients without enhancing tumor (both masks empty).

Three scenarios:
  null    M4 = M1 = M5 = M6 (what the seed-0 validation runs suggest). Only M0 is worse under degradation.
  effect  M4 beats M1 (0.02 at worst, 0.005 clean), M6 trails M4, M5 trails M4: real effects well above
          the seed noise (SD 0.003).
  tiny    M4 beats M1 by 0.008 at worst: consistent across patients (so Wilcoxon is significant) but the
          seed-to-seed SD is 0.006 here, so 0.008 < 2 x SD = 0.012 and DEC-006 must report it as
          "within training noise" regardless of p.

The planted effects come back from `generate_fake_metrics` (and as `*_truth.json` next to the CSV) so tests
can check that the pipeline recovers them. Never writes to results/raw_metrics_v2.csv.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

OUT_CSV = "data/cache/raw_metrics_v2_fake.csv"  # gitignored (data/cache/)
SEED = 42

# DEC-005: clean + SNR {8, 5, 3} x r {1.0, 0.5} x thickness {1, 5 mm}
CONDITIONS = [("clean", "none", 0.0, 1.0, 1)] + [
    (f"kspace_snr{s}_r{r}_t{t}", "kspace", float(s), r, t) for s in (8, 5, 3) for r in (1.0, 0.5) for t in (1, 5)]
WORST = "kspace_snr3_r0.5_t5"
HD95_CONDS = {"clean", "kspace_snr5_r1.0_t1", WORST}
MODELS = ["M0", "M1", "M4", "M5", "M6"]
SEEDS = [0, 1, 2]
N_PATIENTS, N_LGG, N_LGG_NO_ET = 74, 15, 5

BASE_REGION_DICE = {"wt": 0.89, "tc": 0.83, "et": 0.76}
PATIENT_SD = 0.04            # shared across models -> paired design
PATIENT_MODEL_SD = 0.01      # patient x model interaction (fixed across seeds / conditions)
ROW_NOISE_SD = 0.01
SEED_SD = {"M0": 0.02, "default": 0.003}   # exact sample SD of each model's three seed offsets (a scenario may override "default")
M0_DROP_WORST = 0.34         # pilot: worst kept condition costs a clean-trained model 0.34
M0_CLEAN_ADVANTAGE = 0.01    # clean-trained model is slightly better on clean scans (Step C)
AUG_DROP_WORST = 0.07        # augmentation-trained models lose ~0.07 clean -> worst (Step C: 0.825 -> 0.755)

# scenario -> planted M4-over-M1 gain, M4-over-M6 gap and M4-over-M5 gap, each (clean, worst), interpolated by severity;
# optional seed_sd overrides the seed-to-seed SD of the non-M0 models
SCENARIOS = {
    "null":   dict(m4_gain=(0.0, 0.0),     m6_gap=(0.0, 0.0),     m5_gap=(0.0, 0.0)),
    "effect": dict(m4_gain=(0.005, 0.020), m6_gap=(0.003, 0.015), m5_gap=(0.005, 0.005)),
    "tiny":   dict(m4_gain=(0.002, 0.008), m6_gap=(0.0, 0.0),     m5_gap=(0.0, 0.0), seed_sd=0.006),
}


def seed_sd(model: str, scenario: str) -> float:
    return SEED_SD["M0"] if model == "M0" else SCENARIOS[scenario].get("seed_sd", SEED_SD["default"])


def severity(snr: float, r: float, thick: int) -> float:
    """0 for clean; SNR 8 / r 1 / 1 mm = 0.5 (costs M0 0.17); worst (SNR 3, r 0.5, 5 mm) = 1."""
    if snr == 0.0:
        return 0.0
    s_snr = (1 / snr - 1 / 8) / (1 / 3 - 1 / 8)
    return 0.5 + 0.5 * (0.5 * s_snr + 0.25 * (1 - r) / 0.5 + 0.25 * (thick - 1) / 4)


def _interp(pair, sev: float) -> float:
    return pair[0] + (pair[1] - pair[0]) * sev


def model_level(model: str, sev: float, scenario: str) -> float:
    """Planted mean-Dice level of a model at severity `sev`, before the empty-ET scaling."""
    sc = SCENARIOS[scenario]
    if model == "M0":
        return M0_CLEAN_ADVANTAGE - M0_DROP_WORST * sev
    m1 = -AUG_DROP_WORST * sev
    if model == "M1":
        return m1
    m4 = m1 + _interp(sc["m4_gain"], sev)
    if model == "M4":
        return m4
    if model == "M5":
        return m4 - _interp(sc["m5_gap"], sev)
    if model == "M6":
        return m4 - _interp(sc["m6_gap"], sev)
    raise ValueError(model)


def effect_scale() -> float:
    """Planted effects hit WT/TC/ET equally except the empty-ET patients (ET fixed at 1.0): 2/3 of the effect."""
    return ((N_PATIENTS - N_LGG_NO_ET) + N_LGG_NO_ET * 2 / 3) / N_PATIENTS


def planted_truth(scenario: str) -> Dict:
    """Expected population effects on mean Dice, in the units the pipeline reports."""
    k = effect_scale()
    sev = {c[0]: severity(c[2], c[3], c[4]) for c in CONDITIONS}
    lvl = lambda m, c: model_level(m, sev[c], scenario) * k
    drop = lambda m: lvl(m, "clean") - lvl(m, WORST)
    return {
        "scenario": scenario, "scale_empty_et": k,
        "seed_sd_default": seed_sd("M1", scenario), "seed_sd_m0": SEED_SD["M0"],
        "m4_minus_m1_worst": lvl("M4", WORST) - lvl("M1", WORST),
        "m4_minus_m1_clean": lvl("M4", "clean") - lvl("M1", "clean"),
        "h2_drop_difference": drop("M1") - drop("M4"),               # positive = M4 loses less
        "m4_minus_m6_worst": lvl("M4", WORST) - lvl("M6", WORST),
        "m4_minus_m0_worst": lvl("M4", WORST) - lvl("M0", WORST),
        "m4_minus_m0_clean": lvl("M4", "clean") - lvl("M0", "clean"),
        "m5_minus_m4_worst": lvl("M5", WORST) - lvl("M4", WORST),
        "m5_minus_m4_clean": lvl("M5", "clean") - lvl("M4", "clean"),
        "n_empty_et": N_LGG_NO_ET,
    }


def generate_fake_metrics(seed: int = SEED, scenario: str = "null") -> Tuple[pd.DataFrame, Dict]:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {list(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    pids = np.array([f"BraTS20_Training_{i:03d}" for i in range(1, N_PATIENTS + 1)])
    grade = np.array(["HGG"] * (N_PATIENTS - N_LGG) + ["LGG"] * N_LGG)
    empty_et = np.zeros(N_PATIENTS, bool)
    empty_et[N_PATIENTS - N_LGG: N_PATIENTS - N_LGG + N_LGG_NO_ET] = True      # the first 5 LGG

    patient_eff = rng.normal(0, PATIENT_SD, N_PATIENTS)
    pm_eff = {m: rng.normal(0, PATIENT_MODEL_SD, N_PATIENTS) for m in MODELS}
    seed_off = {m: rng.permutation(np.array([-1.0, 0.0, 1.0]) * seed_sd(m, scenario)) for m in MODELS}
    vol_gt = {k: np.exp(rng.normal(np.log(v), 0.6, N_PATIENTS)) for k, v in {"wt": 80.0, "tc": 28.0, "et": 12.0}.items()}
    vol_gt["et"][empty_et] = 0.0

    frames = []
    for m in MODELS:
        for si, s in enumerate(SEEDS):
            for cid, noise, snr, r, thick in CONDITIONS:
                sev = severity(snr, r, thick)
                shared = model_level(m, sev, scenario) + patient_eff + pm_eff[m] + seed_off[m][si]
                dice = {reg: np.clip(base + shared + rng.normal(0, ROW_NOISE_SD, N_PATIENTS), 0.0, 1.0)
                        for reg, base in BASE_REGION_DICE.items()}
                dice["et"] = np.where(empty_et, 1.0, dice["et"])                # BraTS: both empty -> 1
                vol_pred = {reg: np.where(vol_gt[reg] == 0, 0.0, np.maximum(
                    vol_gt[reg] * (1 + rng.normal(0, 0.15 + 0.2 * (1 - dice[reg]))), 0)) for reg in dice}
                f = pd.DataFrame({
                    "model": m, "seed": s, "patient_id": pids, "grade": grade, "condition_id": cid,
                    "noise": noise, "snr": snr, "r": r, "thick": thick,
                    "dice_wt": dice["wt"], "dice_tc": dice["tc"], "dice_et": dice["et"],
                    "dice_mean": (dice["wt"] + dice["tc"] + dice["et"]) / 3.0})
                for reg in ("wt", "tc", "et"):
                    f[f"vol_pred_{reg}_cm3"], f[f"vol_gt_{reg}_cm3"] = vol_pred[reg], vol_gt[reg]
                f["n_components_wt"] = 1 + rng.poisson(0.3 + 4 * (1 - dice["wt"]))
                f["false_tumor_cm3"] = rng.exponential(0.2 + 4 * (1 - dice["wt"]), N_PATIENTS)
                for reg in ("wt", "tc", "et"):
                    if cid in HD95_CONDS:
                        h = np.exp(rng.normal(np.log(4.0 + 6.0 * sev), 0.7, N_PATIENTS)) * (1.2 - dice[reg])
                        f[f"hd95_{reg}"] = np.where(empty_et, 0.0, h) if reg == "et" else h   # both empty -> 0 (BraTS)
                    else:
                        f[f"hd95_{reg}"] = np.nan
                frames.append(f)
    return pd.concat(frames, ignore_index=True), planted_truth(scenario)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="null")
    args = ap.parse_args()
    out = Path(args.out)
    if out.name == "raw_metrics_v2.csv":
        raise SystemExit("Refusing to write fake data to a file named raw_metrics_v2.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df, truth = generate_fake_metrics(scenario=args.scenario)
    df.to_csv(out, index=False)
    out.with_name(out.stem + "_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {len(df)} fake rows ({args.scenario} scenario) to {out}")


if __name__ == "__main__":
    main()
