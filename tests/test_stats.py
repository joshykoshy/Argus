"""Run the DEC-006 stats pipeline on fake data with planted effects and check it recovers them."""

import numpy as np
import pandas as pd
import pytest

from python.analysis import stats
from python.analysis.make_fake_metrics import HD95_CONDS, MODELS, SCENARIOS, generate_fake_metrics
from python.analysis.stats import CLEAN, MID, WORST, bootstrap_ci, holm, paired_wilcoxon


@pytest.fixture(scope="module", params=list(SCENARIOS))
def scen(request, tmp_path_factory):
    df, truth = generate_fake_metrics(scenario=request.param)
    tmp = tmp_path_factory.mktemp(request.param)
    csv = tmp / "fake.csv"
    df.to_csv(csv, index=False)
    res = stats.run_statistical_tests(str(csv), str(tmp / "tables"))
    return request.param, df, truth, res, tmp


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    df, truth = generate_fake_metrics(scenario="null")
    return df, truth


def get(res, name):
    return res["hypotheses"].set_index("hypothesis").loc[name]


def sec(res, test, ref, cond):
    t = res["secondary_tests"]
    r = t[(t["test_model"] == test) & (t["ref_model"] == ref) & (t["condition"] == cond)]
    assert len(r) == 1
    return r.iloc[0]


# ---- fake data and helpers ---------------------------------------------------------------------
def test_fake_data_matches_evaluator_schema(null):
    from evaluation.evaluate_v2 import COLUMNS, HD95_CONDS as EVAL_HD
    df, truth = null
    assert list(df.columns) == COLUMNS
    assert set(EVAL_HD) == HD95_CONDS
    assert len(df) == 74 * 13 * 5 * 3 and set(df["model"]) == set(MODELS)
    assert df["condition_id"].nunique() == 13 and WORST in set(df["condition_id"])


def test_hd95_only_at_three_conditions_and_both_empty_is_zero(null):
    df, truth = null
    has = df[["hd95_wt", "hd95_tc", "hd95_et"]].notna().all(axis=1)
    assert set(df[has]["condition_id"]) == HD95_CONDS
    assert not df[~df["condition_id"].isin(HD95_CONDS)][["hd95_wt", "hd95_tc", "hd95_et"]].notna().any().any()
    empty = df[(df["vol_gt_et_cm3"] == 0) & df["condition_id"].isin(HD95_CONDS)]
    assert empty["patient_id"].nunique() == truth["n_empty_et"] and (empty["hd95_et"] == 0.0).all()


def test_holm_matches_hand_example():
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_bootstrap_is_seeded():
    x = np.random.default_rng(0).normal(size=50)
    assert bootstrap_ci(x) == bootstrap_ci(x)


def test_wilcoxon_all_zero_gives_p1():
    assert paired_wilcoxon(np.zeros(10))[1] == 1.0


# ---- DEC-006 structure -------------------------------------------------------------------------
def test_outputs_written(scen):
    _, _, _, _, tmp = scen
    for name in ["hypotheses", "secondary_tests", "dice_curves", "headline_metrics", "seed_noise_floor"]:
        assert (tmp / "tables" / f"{name}.csv").exists()


def test_hypotheses_are_the_registered_ones(scen):
    _, _, _, res, _ = scen
    h = res["hypotheses"].set_index("hypothesis")
    assert list(h.index) == ["H1", "H2", "H3"]
    assert (h.loc["H1", "test_model"], h.loc["H1", "ref_model"], h.loc["H1", "condition"]) == ("M4", "M1", WORST)
    assert (h.loc["H2", "test_model"], h.loc["H2", "ref_model"], h.loc["H2", "kind"]) == ("M4", "M1", "drop")
    assert (h.loc["H3", "test_model"], h.loc["H3", "ref_model"], h.loc["H3", "condition"]) == ("M4", "M6", WORST)
    assert (h["n_patients"] == 74).all()


def test_holm_over_the_three(scen):
    _, _, _, res, _ = scen
    h = res["hypotheses"]
    assert np.allclose(h["p_holm"], holm(h["raw_p"].to_numpy()))
    assert (h["p_holm"] >= h["raw_p"] - 1e-12).all()


def test_seed_noise_floor_recovered(scen):
    _, _, truth, res, _ = scen
    f = res["seed_noise_floor"]
    f = f[(f["metric"] == "dice_mean") & (f["condition"] == CLEAN)].set_index("model")["seed_sd"]
    assert f["M0"] == pytest.approx(truth["seed_sd_m0"], abs=0.003)
    for m in ["M1", "M4", "M5", "M6"]:
        assert f[m] == pytest.approx(truth["seed_sd_default"], abs=0.002)


def test_secondary_reports_m4_vs_m0_and_m5_vs_m4(scen):
    _, _, truth, res, _ = scen
    r = sec(res, "M4", "M0", WORST)
    assert r["mean_diff"] == pytest.approx(truth["m4_minus_m0_worst"], abs=0.01)
    assert r["raw_p"] < 1e-6 and not r["within_training_noise"]
    assert sec(res, "M4", "M0", CLEAN)["mean_diff"] == pytest.approx(truth["m4_minus_m0_clean"], abs=0.01)
    assert sec(res, "M5", "M4", WORST)["mean_diff"] == pytest.approx(truth["m5_minus_m4_worst"], abs=0.006)


# ---- planted scenarios -------------------------------------------------------------------------
def test_null_scenario_supports_nothing(scen):
    name, _, truth, res, _ = scen
    if name != "null":
        pytest.skip("null scenario only")
    h = res["hypotheses"]
    assert (h["verdict"] == "NOT SUPPORTED").all()
    assert h["within_training_noise"].all()
    assert (h["mean_diff"].abs() < 0.006).all()


def test_effect_scenario_supports_all_three(scen):
    name, _, truth, res, _ = scen
    if name != "effect":
        pytest.skip("effect scenario only")
    h = res["hypotheses"].set_index("hypothesis")
    assert (h["verdict"] == "SUPPORTED").all()
    assert not h["within_training_noise"].any()
    assert h.loc["H1", "mean_diff"] == pytest.approx(truth["m4_minus_m1_worst"], abs=0.005)
    assert h.loc["H2", "mean_diff"] == pytest.approx(truth["h2_drop_difference"], abs=0.006)
    assert h.loc["H3", "mean_diff"] == pytest.approx(truth["m4_minus_m6_worst"], abs=0.005)
    for n in ("H1", "H2", "H3"):
        assert h.loc[n, "ci95_low"] - 0.003 <= {"H1": truth["m4_minus_m1_worst"], "H2": truth["h2_drop_difference"],
                                                 "H3": truth["m4_minus_m6_worst"]}[n] <= h.loc[n, "ci95_high"] + 0.003


def test_tiny_effect_is_significant_but_within_training_noise(scen):
    name, _, truth, res, _ = scen
    if name != "tiny":
        pytest.skip("tiny scenario only")
    h1 = get(res, "H1")
    assert h1["p_holm"] < 0.05 and h1["mean_diff"] > 0          # Wilcoxon sees it ...
    assert h1["within_training_noise"]                          # ... but it is below 2 x seed SD
    assert h1["verdict"] == "NOT SUPPORTED"
    assert "within training noise" in h1["note"]
    assert h1["mean_diff"] == pytest.approx(truth["m4_minus_m1_worst"], abs=0.004)
    assert h1["seed_noise_floor"] == pytest.approx(0.006, abs=0.001)


# ---- direction, alignment, curves --------------------------------------------------------------
def test_opposite_direction_is_not_supported(null):
    df, _ = null
    df = df.copy()
    m = (df["model"] == "M4") & (df["condition_id"] == WORST)
    for c in ["dice_wt", "dice_tc", "dice_et", "dice_mean"]:
        df.loc[m, c] = df.loc[m, c] - 0.05                         # M4 clearly worse than M1 and M6
    pl = stats.patient_level(df)
    h = stats.evaluate_hypotheses(pl, df, stats.seed_noise_floor(df)).set_index("hypothesis")
    assert h.loc["H1", "verdict"] == "NOT SUPPORTED" and "OPPOSITE" in h.loc["H1", "note"]
    assert h.loc["H1", "mean_diff"] < 0


def test_alignment_by_patient_id_not_position(null):
    df, _ = null
    pl, floor = stats.patient_level(df), stats.seed_noise_floor(df)
    before = stats.evaluate_hypotheses(pl, df, floor)
    shuffled = pl.sample(frac=1.0, random_state=0).reset_index(drop=True)
    after = stats.evaluate_hypotheses(shuffled, df, floor)
    assert np.allclose(before["mean_diff"], after["mean_diff"])
    partial = pl[~((pl["model"] == "M6") & (pl["patient_id"] == pl["patient_id"].iloc[0]))]
    h3 = stats.evaluate_hypotheses(partial, df, floor).set_index("hypothesis").loc["H3"]
    assert h3["n_patients"] == 73                              # unmatched patient dropped, rest still paired by id


def test_dice_curves_cover_13_conditions_with_bootstrap_cis(scen):
    _, _, _, res, _ = scen
    c = res["dice_curves"]
    d = c[c["metric"] == "dice_mean"]
    assert len(d) == 5 * 13 and (d["n_patients"] == 74).all()
    assert ((d["ci95_low"] <= d["mean"]) & (d["mean"] <= d["ci95_high"])).all()
    m0 = d[d["model"] == "M0"].set_index("condition")["mean"]
    assert m0[CLEAN] - m0[WORST] > 0.25                        # clean-trained model collapses at the worst condition
    assert {"snr", "r", "thick", "noise"} <= set(c.columns)


def test_headline_hd95_uses_median_and_has_the_three_conditions(scen):
    _, df, _, res, _ = scen
    h = res["headline_metrics"]
    assert set(h["condition"]) == {CLEAN, MID, WORST}
    r = h[(h["model"] == "M4") & (h["condition"] == WORST)].iloc[0]
    assert r["hd95_et_median"] > 0 and r["hd95_et_q25"] <= r["hd95_et_median"] <= r["hd95_et_q75"]


def test_missing_model_or_columns_raise(null, tmp_path):
    df, _ = null
    bad = df[df["model"] != "M6"]
    bad.to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(ValueError, match="models missing"):
        stats.run_statistical_tests(str(tmp_path / "a.csv"), str(tmp_path / "t"))
    df.drop(columns=["thick"]).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        stats.run_statistical_tests(str(tmp_path / "b.csv"), str(tmp_path / "t"))
