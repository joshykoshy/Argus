# Roadmap to the finished paper

Legend: [x] done · [ ] to do · **(you)** = MATLAB side · **(team)** = Python side

The paper's single claim: *when MRI scans get worse (noisier, blurrier, like a cheap
low-field scanner), does our model (M4) lose less accuracy than a normal U-Net (M0)?*
Plus: *do the tumor measurements a doctor would use stay correct?*

---

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
- [ ] 1.3 **3D renderer** - see-through brain with colored tumor parts, truth vs
      prediction side by side, plus a heatmap painting where the prediction is off
      and by how many mm. Tested on the damaged masks.
      -> `figures/render3d_*.png`
- [ ] 1.4 **Diagnostic report (4 of 5 panels)** - pick 5 example patients by a fixed
      rule using only true outlines (small to large tumors, both HGG and LGG, one
      with no enhancing tumor). Panels: original, degraded, truth, band split.
      Panel 4 (prediction) waits for Stage 3.
      -> `figures/diagnostic_*.png`, selection rule written down
- [ ] 1.5 **Write the paper parts that don't need results** - introduction (why
      low-field MRI matters), dataset and splits, how we simulate bad scans,
      the frequency split at D0=0.20, MATLAB-Python check, spectral analysis.

## Stage 2 - Teammate finishes the models (team, in parallel with Stage 1)
- [x] 2.1 M0-M4 trained
- [ ] 2.2 M5-M7 trained
- [ ] 2.3 3D test evaluation: 8 models x 74 patients x 13 scan conditions
      -> `results/predictions/*.nii.gz` + `results/raw_metrics.csv`
- [ ] 2.4 Settle the open questions with the teammate (raise early, they shape
      what we can claim):
  - [ ] Was M4 trained on clean scans only, or with degraded ones like M1?
  - [ ] H3 names the wrong comparison model (M6 still uses the split) - fix wording
  - [ ] Can M0-M4 be evaluated now, before M5-M7 finish, to save time?

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

## Stage 4 - Statistics and verdicts (team runs `python/analysis/stats.py`; you check)
- [ ] 4.1 **H1**: does M4 beat M0 by >= 0.05 Dice at SNR=8?
- [ ] 4.2 **H2**: does M4 lose significantly less Dice from clean to SNR=8 than M0?
      (Wilcoxon test, Holm correction for multiple tests)
- [ ] 4.3 **H3**: does the frequency split alone explain >= 50% of M4's gain?
- [ ] 4.4 Final tables + figures (Dice vs condition curves, HD95, false positives)

Each verdict can be "supported", "not supported", or "mixed". All three are
publishable if reported honestly; early validation data hints M4 may *not* be
more robust, so plan for that.

## Stage 5 - Finish the paper
- [ ] 5.1 Results section (numbers from Stages 3-4)
- [ ] 5.2 Discussion (what it means, limitations, why M4 did/didn't help)
- [ ] 5.3 Abstract and conclusion (written last)
- [ ] 5.4 Figure/table polish, references, proofread
- [ ] 5.5 Selective push of your final files to `team/matlab-analysis`
