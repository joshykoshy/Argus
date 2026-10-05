# Task brief: Python-side fixes and the Phase 6 evaluation (Mayank)

Paste this whole file into your AI coding assistant, opened in your local
clone of `https://github.com/mayankt411/Argus`.

---

## Context

You are working on the ECTE408 study: does splitting MRI into low- and
high-frequency bands before a U-Net (M4, dual-stream, D0 = 0.20) make brain
tumor segmentation more robust to simulated low-field MRI degradation?
BraTS 2020, fixed split `data/splits/split_v1.json` (258 / 37 / 74, seed 42),
13 degradation conditions, 8 models M0-M7, all trained with seed 0 for
5 epochs.

A teammate (Joshua, MATLAB / morphometrics side) audited the Python code
while building the MATLAB analysis. His full write-up, with file and line
evidence for each point, is `docs/LAB_NOTEBOOK.md` -> "Critical findings for
the team" on his branch `matlab-analysis`. The items below are the ones that
need Python changes. Verify each claim in the code yourself before changing
anything.

## Hard rules

- Never change `data/splits/split_v1.json`.
- Do not edit anything under `matlab/` (Joshua's area).
- Keep the prediction output format **exactly** as now, because the MATLAB
  pipeline is already built and tested against it:
  `results/predictions/{model}_{patient_id}_{condition}.nii.gz`, uint8,
  labels 0 / 1 = NCR / 2 = edema / 3 = ET, full 240 x 240 x 155 with the
  original affine; `results/predictions/ground_truth/GT_{patient_id}.nii.gz`;
  conditions `clean`, `snr12_r0.75`, `snr8_r0.5`.
- Never commit `*.nii`, `*.nii.gz`, `*.npz` or `results/predictions/`.
- Work on a branch; open a PR into `master`.

## Tasks, in priority order

### P0-1. Fix HD95 on empty masks (finding C9) - before any evaluation

`python/metrics/evaluation.py`, `compute_3d_hd95`: currently
`if np.sum(pred_b) == 0 or np.sum(target_b) == 0: return 373.13`.
When **both** masks are empty (patient has no enhancing tumor and the model
correctly predicts none), this returns the maximum penalty. The BraTS
convention is 0 for that case. Five test patients have no ET (BraTS20_Training
266, 279, 281, 310, 335), so mean ET HD95 is inflated by ~25 mm for every model.

Fix: return 0.0 when both are empty; keep 373.13 when exactly one is empty.
Add a unit test in `tests/` covering all three cases (both empty, one empty,
neither). Dice already handles this correctly; do not change it.

### P0-2. Three model pairs are identical in code (finding C1) - decide and fix

In `python/models/factory.py` only five distinct models exist:

| Model | README says | Code builds | Same as |
|---|---|---|---|
| M2 | NLM denoiser + U-Net | plain `UNetSingle` (line ~97) | M1 |
| M3 | low-pass + CLAHE + U-Net | `M3ModelWrapper.forward` = low band only, no CLAHE | M5 |
| M7 | dual-stream with enhanced high band | `M7ModelWrapper.forward` = same as M4 | M4 |

`equalize_adapthist` and `white_tophat` are imported but never called. The
training logs confirm it: M3 vs M5 epoch-1 loss 1.48710 vs 1.48686, M4 vs M7
1.50330 vs 1.50323.

For each of M2, M3, M7, either:
- (a) implement as described, then retrain only that model:
  - M2: non-local-means denoising of each input channel before the U-Net
    (`skimage.restoration.denoise_nl_means`, h tied to the estimated noise
    sigma; the tuning script's `nlm_h = 0.10` was never actually used with
    NLM, see `python/tune_cutoff.py:135-140`). NLM is slow on CPU; consider
    applying it in the dataset / evaluation pipeline once per slice rather
    than inside `forward`.
  - M3: low band, then CLAHE (`equalize_adapthist`) on the T1ce and FLAIR
    channels (rescale to [0, 1] first).
  - M7: dual-stream where the high band is cleaned with morphology
    (e.g. white top-hat) before stream B.
- (b) or drop it and remove it from the README, configs, evaluation and the
  paper's model table.

Acceptance: a test that builds every model and asserts no two models give
identical outputs on the same random input (set seeds; compare forward
passes).

Note: M5 does not need retraining (it is correct as "low band only"); M4 is
correct.

### P0-3. Statistics hypotheses do not match the study plan (finding C10)

`python/analysis/stats.py` tests different hypotheses than the plan and the
paper draft:

| | Plan / paper | `stats.py` |
|---|---|---|
| H1 | M4 beats M0 by >= 0.05 mean Dice at SNR 8 | M0 drops by > 0.15 |
| H2 | M4's clean-to-SNR8 drop is smaller than M0's (paired Wilcoxon, p < 0.05) | M1 recovers > 0.10 over M0 (no test) |
| H3 | M5 explains >= 50 % of M4's gain over M6 | M4 significantly beats M1 |

Agree with the team which set is official and write it down in
`docs/DECISIONS.md` **before** any results are seen. Nazir is rewriting
`stats.py` to match (his own brief); you only need to make the decision.
Also note: plan-H3 names M6 as "M4 without decomposition", but M6 (early
fusion) also uses both bands; restate it.

### P1-1. More seeds (finding C7)

`configs/base.yaml` says `seeds: [0, 1, 2]` and 20 epochs; every model was
trained with seed 0 and 5 epochs. The duplicate pairs above measure the
run-to-run noise directly: identical code and seed, final mean validation
Dice still differs by up to 0.011 and clean WT Dice by up to 0.015. The
whole M0 robustness drop is ~0.006. Without more seeds no model difference
under ~0.02 Dice can be claimed.

Minimum: seeds 1 and 2 for M0, M1 and M4 (the models the hypotheses use).
Ideally all models. `python/evaluate.py` currently hard-codes `seeds=[0]`;
include the extra seeds in `raw_metrics.csv` (NIfTIs for seed 0 only is fine).
Update `docs/DEVIATIONS.md` with the epochs and seeds actually used.

### P1-2. Team decisions that would require retraining everything

Discuss with the team before doing either; if you do, do it before P1-1.

- **C2, degradation too mild.** `results/m0_degradation_metrics.csv`: M0
  (clean-trained) loses 0.0058 mean Dice from clean to SNR 8 / r 0.5. Options:
  extend the range (e.g. SNR 5 and 3; through-plane resolution loss, since
  real low-field scans have thick slices), or keep it and frame the paper as
  a negative / null result.
- **C4, noise order.** `python/physics/degradation.py:91-121` adds noise in
  image space after k-space truncation, so noise refills the removed
  frequencies. A real low-resolution acquisition has noise only in the
  acquired band. Fix: add complex noise to K_trunc inside the mask, then
  inverse-transform and take the magnitude. If changed, the MATLAB port
  `matlab/reference/degrade_slice.m` and the parity test must be updated
  together (tell Joshua).

### P2. Documentation corrections

- `docs/DECISIONS.md` DEC-004: the D0 proxy search (Dice ~0.03 on 3 val
  patients) is not evidence; describe D0 = 0.20 as a design choice
  (finding C5).
- `docs/FINDINGS_SPECTRAL.md` is hard-coded `fprintf` text in
  `matlab/spectral_analysis.m`, not computed. The "high band below 0 dB at SNR
  8-12" claim is false (measured +1.0 to +5.6 dB, `matlab/spectral_check.m` on
  Joshua's branch). Mark the file as superseded (finding C8).
- `docs/FINDINGS_SPECTRAL.md` and elsewhere: SNR is a linear ratio
  (sigma = mu_brain / SNR), not dB.

### P3. Phase 6: full 3D evaluation

Only after P0-1 and P0-2 (and P1 if adopted):

```
python -m python.evaluate
```

Then:
1. Confirm `results/predictions/` holds all
   `models x 74 patients x 3 conditions` files plus `ground_truth/GT_*.nii.gz`
   (74). The MATLAB side refuses to run on a partial folder.
2. Commit `results/raw_metrics.csv` (small) to `master`.
3. Share `results/predictions/` with Joshua outside git (zip, cloud drive).
4. Tell Joshua and Nazir it is complete.

## Deliverables checklist

- [ ] PR: HD95 fix + test
- [ ] PR: M2 / M3 / M7 implemented or removed + distinctness test
- [ ] `docs/DECISIONS.md`: official H1-H3 wording, D0 as design choice
- [ ] extra seeds trained (at least M0, M1, M4 x seeds 1, 2)
- [ ] team decision recorded on C2 / C4
- [ ] `docs/DEVIATIONS.md` updated
- [ ] Phase 6 complete: `raw_metrics.csv` committed, predictions shared
