# Task brief: interactive 3D viewer, statistics pipeline, literature (Nazir)

Paste this whole file into your AI coding assistant, opened in your local
clone of `https://github.com/mayankt411/Argus`. None of this waits for
Mayank's retraining: everything is built and tested now on ground truth and
synthetic data, then pointed at the real outputs when they arrive.

---

## Context

The ECTE408 study asks whether splitting MRI into low- and high-frequency
bands before a U-Net (M4, dual-stream) makes brain tumor segmentation more
robust to simulated low-field MRI. Data: BraTS 2020 (Kaggle
`awsaf49/brats20-dataset-training-validation`), fixed split
`data/splits/split_v1.json` (74 test patients). Regions: WT = labels 1+2+4,
TC = 1+4, ET = 4 (raw BraTS files use 4 for ET; the Python predictions use 3;
always accept both).

Three people:
- **Mayank** - Python training / evaluation (`python/`), currently fixing
  models and retraining.
- **Joshua** - MATLAB morphometrics, agreement analysis, 3D renders,
  paper Methods. His work and full audit: branch `matlab-analysis`,
  `docs/LAB_NOTEBOOK.md` (read "Critical findings" and "Picture tour") and
  `docs/PAPER_DRAFT.md`.
- **You** - the three tasks below.

Get Joshua's branch first (ask him whether it is on the team repo yet;
otherwise add his fork as a remote):
```
git fetch origin matlab-analysis     # if pushed to the team repo
# or: git remote add joshua https://github.com/joshykoshy/Argus.git && git fetch joshua
```

## Hard rules

- Never change `data/splits/split_v1.json`.
- Do not edit `matlab/` (Joshua) or `python/models`, `python/data`,
  `python/physics`, `python/evaluate.py`, `python/train*.py` (Mayank).
- Your areas: new folder `viz3d/`, `python/analysis/` (coordinate with
  Mayank, who wrote it), `tests/` for your own tests, `docs/paper/`.
- Never commit `*.nii`, `*.nii.gz`, `*.npz`, `results/predictions/`.
- One configurable data-path variable at the top of each script; no
  hard-coded absolute paths.
- Fixed random seeds for anything random.
- Work on your own branch (e.g. `nazir-analysis`); PR into `master`.

---

## Task 1 - Interactive 3D tumor viewer (`viz3d/`)

**Goal:** a standalone HTML page per patient that anyone can open in a
browser and rotate: see-through brain, tumor regions in colour, and (once
predictions exist) truth vs prediction with the prediction's surface
coloured by error. For presentations, supplementary material and teammates
without MATLAB. Joshua's MATLAB renderer (`matlab/render_tumor_3d.m`) makes
the static paper figures; yours is the interactive companion and must use
the **same conventions** so the two agree.

**Build:**
1. `viz3d/meshes.py`
   - load a label NIfTI (nibabel), make WT / TC / ET masks (ET = 3 or 4)
   - mesh each with marching cubes (`skimage.measure.marching_cubes`) after
     Gaussian smoothing sigma = 0.5 voxel at level 0.5 (same as MATLAB)
   - brain surface from FLAIR > 0 (downsample 2x, sigma 1.5, decimate)
   - vertices in mm using the header voxel spacing
   - optional STL export (`trimesh`)
2. `viz3d/surface_error.py`
   - signed distance to the true WT boundary, in mm:
     `sdf = distance_transform_edt(~truth) - distance_transform_edt(truth)`
     (positive outside, negative inside), sampled at every predicted-surface
     vertex with trilinear interpolation
   - summary: mean signed, mean absolute, 95th-percentile absolute,
     fraction of vertices > 2 mm
3. `viz3d/viewer.py` - plotly `Mesh3d`, writes
   `figures/interactive/<patient>_<label>.html` (`include_plotlyjs='cdn'` to
   keep files small):
   - truth-only mode: brain (opacity ~0.1), WT blue (0.25), TC orange
     (0.35), ET red (1.0); legend toggles each layer
   - truth-vs-prediction mode: two scenes side by side, plus the predicted
     WT surface coloured by signed error on a blue-white-red scale, fixed
     range -5 to +5 mm, colourbar labelled
     "signed distance to true boundary (mm): - inside, + outside"
4. **Validation (required, `tests/test_viz3d.py`):** build synthetic
   "predictions" by damaging the ground truth, exactly as Joshua did
   (`matlab/damage_mask.m`: Euclidean dilation/erosion via distance
   transform, shift, etc.), and assert:
   - truth vs itself: |mean signed| < 0.2 mm, mean abs < 0.5 mm
   - dilate 2 mm: mean signed = +2.0 +- 0.2 mm
   - erode 2 mm: mean signed = -2.0 +- 0.2 mm
   Joshua's MATLAB values for patient BraTS20_Training_002 are -0.03 / 0.20,
   +2.04, -2.05 mm (`results/render3d_surface_error.csv` on his branch); yours
   should be within ~0.1 mm of these.
5. Generate truth-only viewers now for the 5 example patients
   (`results/diagnostic_selection.csv` on Joshua's branch: 008, 167, 236,
   292, 281) and one truth-vs-"dilate 2 mm" demo. When predictions arrive,
   generate M0 vs M4 at `snr8_r0.5` for the same 5.

## Task 2 - Statistics pipeline that matches the study (`python/analysis/`)

**Problem (finding C10):** `python/analysis/stats.py` tests different
hypotheses from the study plan and the paper draft:

| | Plan / paper | current `stats.py` |
|---|---|---|
| H1 | M4 beats M0 by >= 0.05 mean Dice at SNR 8 | M0 drops by > 0.15 |
| H2 | M4's clean-to-SNR8 drop smaller than M0's (paired Wilcoxon, p < 0.05) | M1 recovers > 0.10 over M0, no test |
| H3 | M5 explains >= 50 % of M4's gain over M6 | M4 significantly beats M1 |

Mayank and the team decide which wording is official (recorded in
`docs/DECISIONS.md`). Build the pipeline so the hypothesis definitions are
one clearly separated, easily edited block.

**Build:**
1. `python/analysis/make_fake_metrics.py` - writes a synthetic
   `raw_metrics.csv` with the exact columns `python/evaluate.py` produces
   (model, seed, patient_id, grade, condition_id, snr, r, dice_wt/tc/et/mean,
   hd95_wt/tc/et, vol_* ...), 74 patients x 13 conditions x 8 models x 3 seeds,
   with **known planted effects** (e.g. M4 = M0 + 0.03 at SNR 8, M0 drop
   0.08, seed noise SD 0.01, 5 LGG patients with empty ET).
2. Extend / rewrite `stats.py`:
   - per-patient paired Wilcoxon signed-rank (seed-averaged per patient),
     Holm correction over a **pre-declared** family of comparisons; report
     effect size (median paired difference + bootstrap 95 % CI, seed 42)
   - the official H1-H3 as functions returning verdict, estimate, CI, p
   - the clean-to-degraded **drop** per patient as its own variable (H2)
   - always include **M4 vs M1** (same training data, different
     architecture; M4 vs M0 mixes architecture with augmentation, finding C3)
   - **seed noise floor:** SD across seeds of each model's mean Dice; flag any
     model difference smaller than 2x that floor as "within training noise"
   - HD95: report **median** and IQR, not mean (373 mm penalty values)
   - skip comparisons between models that are identical in code, if Mayank
     has not fixed them (finding C1: M2 = M1, M3 = M5, M7 = M4)
3. `tests/test_stats.py`: run the pipeline on the fake data and assert it
   recovers the planted effects (sign, size within CI, significance where
   planted, "within noise" where planted below the floor).
4. When the real `results/raw_metrics.csv` lands: run it, produce
   `tables/statistical_tests.csv`, `tables/headline_metrics.csv`,
   `tables/hypotheses.csv` and a short `docs/RESULTS_STATS.md` in plain
   words.

## Task 3 - Literature and Related Work (`docs/paper/`)

1. **Verify every reference** in `docs/PAPER_DRAFT.md` (Joshua's branch,
   list at the end, each marked "verify"): correct authors, title, year,
   venue, DOI. Fix or replace any that are wrong. Find a source for the open
   `[CITE: global MRI access]`.
2. Write **`docs/paper/related_work.md`** (~800-1200 words, every claim
   cited, BibTeX in `docs/paper/references.bib`):
   - low-field / portable MRI and its image-quality trade-offs
   - deep-learning brain tumor segmentation (BraTS, U-Net family)
   - robustness of segmentation to noise / resolution / domain shift
     (augmentation, denoising pre-processing, domain adaptation)
   - frequency-domain and multi-band approaches in medical image networks
   - volumetric / morphometric agreement in tumor assessment
     (Bland-Altman, ICC, RANO-style volumetric response)
   End with 2-3 sentences on the gap this study fills.
3. Only cite papers you have actually opened. If unsure, mark
   `[UNVERIFIED]`, never invent.

---

## Deliverables checklist

- [ ] `viz3d/` + `tests/test_viz3d.py` passing (values match MATLAB)
- [ ] `figures/interactive/` truth-only viewers for 5 patients + 1 demo
- [ ] fake-metrics generator + rewritten `stats.py` + `tests/test_stats.py` passing
- [ ] references verified, `related_work.md`, `references.bib`
- [ ] after real data: tables + `docs/RESULTS_STATS.md`, viewers for M0 vs M4
