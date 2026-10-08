# Roadmap to the finished paper

Legend: [x] done · [ ] to do · **(you)** = MATLAB side · **(team)** = Python side

The paper's single claim: *when MRI scans get worse (noisier, blurrier, like a cheap
low-field scanner), does our model (M4) lose less accuracy than a normal U-Net (M0)?*
Plus: *do the tumor measurements a doctor would use stay correct?*

---

## Redesign (2026-10-06, after the professor meeting)
The degradation is too mild for any model to show a gain (C2), so the training is
being redone by Joshua on Colab Pro, in agreement with Mayank. Order:
- [ ] R0 **Pilot, no training:** run the trained checkpoints on the validation
      patients under a harsher grid (SNR 8/5/3/2, r 1.0/0.5, 5 mm slices, both
      noise orders) -> pick the range where M0 loses >= 0.10 Dice
      (`pilot/pilot_colab.ipynb`)
      -> first run done 2026-10-07: clean 3D Dice ~0.2 for every model; the
         diagnostic showed false tumor on every tumor-free slice (C11), so the
         range cannot be chosen from the v1 models
- [ ] RA **Step A:** retrain M0 on all brain slices with full-3D validation
      (`training/train_colab.ipynb`)
- [ ] RB **Step B:** rerun the pilot with the new M0 -> choose the range
- [ ] Tell Mayank about C11 before he runs the Phase 6 evaluation
- [ ] R1 Fix the noise order (C4) and write the new range + H1-H3 into
      `docs/DECISIONS.md` BEFORE training
- [ ] R2 Train M0, M1, M4, M5, M6 x 3 seeds, all tumor slices + some tumor-free,
      30-50 epochs, on Colab
- [ ] R3 Evaluate on the 74 test patients -> Stage 3 (MATLAB) + Stage 4 (stats)

## Stage 0 - Setup (done)
- [x] 0.1 Fork repo, remotes (`origin` = yours, `team` = shared), branch `matlab-analysis`
- [x] 0.2 Dataset extracted to `archive/`, kept out of git
- [x] 0.3 MATLAB + Image Processing Toolbox confirmed (R2026b)
- [x] 0.4 Parity: MATLAB physics reproduces the reference exactly (`matlab/env_check.m`)
- [x] 0.5 All 74 test patients present, 1 mm voxels, labels {0,1,2,4}

## Stage 1 - Build all your tools using the true tumor outlines (you, now)
Idea: real model outputs don't exist yet, so we test every tool on the *correct
answers* (ground truth). When predictions arrive, we just swap the input folder.

- [x] 1.1 **Ground-truth volumetrics** - for each of the 74 patients and each region
      (whole tumor, tumor core, enhancing tumor): volume in cm3, surface area,
      sphericity (how ball-like), elongation, number of separate pieces.
      Surface measured two ways (voxel count vs smoothed mesh) because blocky
      voxels overestimate surface area.
      -> `results/volumetrics_gt.csv`
- [x] 1.2 **Agreement analysis, validated on fake damage** - deliberately damage the
      true outlines (grow by 1-2 voxels, shrink, shift, cut holes, add fake blobs),
      then check that the comparison tools report exactly the error we caused.
      Tools: volume error %, Bland-Altman plot (shows bias and spread), ICC
      (one number for how well two measurements agree).
      -> `figures/agreement_*.png`, `results/agreement_validation.csv`
- [x] 1.3 **3D renderer** - see-through brain with colored tumor parts, truth vs
      prediction side by side, plus a heatmap painting where the prediction is off
      and by how many mm. Tested on the damaged masks.
      -> `figures/render3d_*.png`
- [x] 1.4 **Diagnostic report (4 of 5 panels)** - pick 5 example patients by a fixed
      rule using only true outlines (small to large tumors, both HGG and LGG, one
      with no enhancing tumor). Panels: original, degraded, truth, band split.
      Panel 4 (prediction) waits for Stage 3.
      -> `figures/diagnostic_*.png`, selection rule written down
- [x] 1.5 **Write the paper parts that don't need results** - introduction (why
      low-field MRI matters), dataset and splits, how we simulate bad scans,
      the frequency split at D0=0.20, MATLAB-Python check, spectral analysis.

## Stage 2 - Teammate finishes the models (team, in parallel with Stage 1)
- [x] 2.1 M0-M4 trained
- [x] 2.2 M5-M7 trained (2026-10-06; C1 duplicates not fixed, see notebook)
- [ ] 2.3 3D test evaluation: 8 models x 74 patients x 13 scan conditions
      -> `results/predictions/*.nii.gz` + `results/raw_metrics.csv`
- [ ] 2.4 Settle the open questions with the teammate (raise early, they shape
      what we can claim):
  - [x] Was M4 trained on clean scans only, or with degraded ones like M1?
        -> answered from code: with degradation (see LAB_NOTEBOOK C3)
  - [ ] Discuss critical findings C1, C2, C4, C5, C7, C8, C9, C10 (LAB_NOTEBOOK.md)
  - [x] Hand out task briefs: `docs/handoff/TASK_MAYANK.md`, `docs/handoff/TASK_NAZIR.md`
  - [ ] H3 names the wrong comparison model (M6 still uses the split) - fix wording
  - [x] ~~Can M0-M4 be evaluated now?~~ moot: all 8 trained

Gate: do not start Stage 3 until **all** predictions are written (a half-full
folder would mix models).

## Stage 3 - Plug real predictions into your tools (you)
- [ ] 3.0 Check the Python pipeline's 192x192 crop didn't cut off any tumor:
      compare `results/predictions/ground_truth/GT_*.nii.gz` volumes against
      `results/volumetrics_gt.csv` (raw files). Any difference = tumor the
      models could never predict.
- [ ] 3.1 Rerun volumetrics on predictions: set `SOURCE = 'pred'` in
      `matlab/volumetrics.m` (NIfTIs exist for clean, snr12_r0.75, snr8_r0.5
      only; the other 10 conditions have volumes in `raw_metrics.csv`)
- [ ] 3.2 Measurement error per model per scan condition: how wrong is the
      predicted tumor volume as the scan gets worse? (the clinical result)
- [ ] 3.3 Agreement (Bland-Altman, ICC) for M0 vs M4 at clean and worst condition
- [ ] 3.4 Diagnostic report panel 4 (prediction overlay) for the 5 patients
- [ ] 3.5 3D comparisons M0 vs M4 with error heatmaps

## Stage 4 - Statistics and verdicts (Nazir; you check)
- [x] 4.0 Statistics pipeline rebuilt to match the paper's H1-H3 (Nazir, 2026-10-06):
      seed-averaged per-patient paired Wilcoxon, Holm correction, bootstrap 95% CIs
      (seed 42), verdict per hypothesis = supported / not supported / inconclusive,
      seed-noise flag (~0.015 Dice), M4 vs M1 always reported, median HD95,
      duplicates M2/M3/M7 skipped. Validated on fake data with planted effects
      and a null case. Waiting for the real `results/raw_metrics.csv`.
- [ ] 4.1 **H1**: does M4 beat M0 by >= 0.05 Dice at SNR=8?
- [ ] 4.2 **H2**: does M4 lose significantly less Dice from clean to SNR=8 than M0?
      (Wilcoxon test, Holm correction for multiple tests)
- [ ] 4.3 **H3**: does the frequency split alone explain >= 50% of M4's gain?
- [ ] 4.4 Final tables + figures (Dice vs condition curves, HD95, false positives)

Each verdict can be "supported", "not supported", or "mixed". All three are
publishable if reported honestly; early validation data hints M4 may *not* be
more robust, so plan for that.

## Interactive 3D viewer (Nazir)
- [x] Plotly HTML viewer: WT/TC/ET meshes (marching cubes, sigma 0.5, mm units),
      truth vs prediction with signed-distance colouring. Validated on damaged
      masks: dilate 2 mm -> +2.04, erode 2 mm -> -2.05 mm, matching the MATLAB
      renderer within 0.1 mm.
- [ ] M0 vs M4 viewers for the 5 example patients (needs predictions)

## Stage 5 - Finish the paper
- [ ] 5.0 Literature: verify every reference, write Related Work (Nazir, pending)
- [ ] 5.1 Results section (numbers from Stages 3-4)
- [ ] 5.2 Discussion (what it means, limitations, why M4 did/didn't help)
- [ ] 5.3 Abstract and conclusion (written last)
- [ ] 5.4 Figure/table polish, references, proofread
- [ ] 5.5 Selective push of your final files to `team/matlab-analysis`
