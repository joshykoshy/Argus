# Task brief: test-set evaluation (Step D) and statistics update (Nazir)

Paste this whole file into your AI coding assistant. Work in a clone of
**`https://github.com/joshykoshy/Argus`, branch `matlab-analysis`** (public):
that branch has all the redesign code; the team repo's `master` does not yet.

---

## What happened since your last brief (read first)

Full detail: `docs/LAB_NOTEBOOK.md` (sections "C11", "Redesign", "Step A/B/C")
and `docs/DECISIONS.md` (DEC-005, DEC-006, DEC-007).

1. **The v1 models were broken on full volumes (finding C11).** They were
   trained and validated on only the largest-tumor slices, so on a full brain
   they drew tumor on every tumor-free slice: true 3D Dice ~0.2, not the ~0.75
   in the training logs. The degradation "barely hurting" (old C2) was an
   artefact of this and is withdrawn.
2. **Redesign:** `training/train_v2.py` trains on all brain slices with
   full-volume 3D validation. A properly trained clean M0 reaches 3D Dice 0.84
   on validation and loses 0.17 at SNR 8, 0.34 at the worst kept condition.
3. **Pre-registered before training (docs/DECISIONS.md):**
   - DEC-005 degradation: k-space noise inside the acquired band (C4 fix);
     13 conditions = clean + SNR {8, 5, 3} x in-plane r {1.0, 0.5} x slice
     thickness {1, 5 mm}; worst = SNR 3, r 0.5, 5 mm.
   - DEC-006 hypotheses (supersede your current stats.py H1-H3, see Task 2).
   - DEC-007 protocol: models M0 (clean) and M1, M4, M5, M6 (degradation
     augmentation), seeds 0, 1, 2 = 15 runs. M2, M3, M7 dropped (duplicates, C1).
4. **Training (Step C) is running now** on Joshua's Kaggle account. Seed-0
   validation at the worst condition: M0 0.512, M1 0.753, M4 0.753, M5 0.756.
   Checkpoints: `train_v3/<model>/seed_<k>/best_model.pt` (state dict under
   `model_state_dict`), ~17 MB each. Joshua will share the folder with you.

**Do not use any GPU quota or files of Joshua's running job.** Use your own
Kaggle or Colab account.

## Hard rules

- Never change `data/splits/split_v1.json`.
- Do not edit `training/`, `pilot/`, `matlab/`, `python/models`, `python/physics`
  (reuse them by import). Your new code goes in **`evaluation/`** and
  `python/analysis/` (stats).
- **Debug on the 37 VALIDATION patients only.** Run on the 74 TEST patients
  exactly once, after all 15 checkpoints exist. Nobody looks at test numbers
  before that, so nothing can be tuned to them.
- Same degraded volume for every model: seed the noise per (patient,
  condition) with `pilot.pilot.seed_for(pid, cond_id)`, as the training's
  validation does, so all model comparisons are paired.
- Never commit `*.nii*`, `*.npz`, `*.npy`, checkpoints, or `results/predictions/`.
- Work on your own branch; push to your fork or open a PR into
  `joshykoshy/Argus:matlab-analysis`.

---

## Task 1: `evaluation/evaluate_v2.py` + a Kaggle/Colab notebook

Reuse, do not reimplement:
- `training/prepare.py` - turns BraTS into memory-mappable arrays
  (`<id>_x.npy` (155,4,192,192) float16, `_y` labels 0/1/2/3, `_m` brain
  mask, `_mu`); run it with `--sets val` while debugging, `--sets test` for
  the final run.
- `training/train_v2.py` - `DEC005` (the 13 condition dicts) and the exact
  inference recipe in `validate()`: `degrade_volume` -> `normalise` ->
  model in slices of 32 -> sigmoid > 0.5 -> WT/TC/ET.
- `pilot/pilot.py` - `degrade_volume`, `normalise`, `dice`, `seed_for`.
- `python/metrics/evaluation.py` - `compute_3d_hd95` **but** fix the BraTS
  convention locally (finding C9): both masks empty -> 0.0; exactly one empty
  -> 373.13.

What it does:
1. Load all available checkpoints (`--ckpt_root`, every
   `<model>/seed_<k>/best_model.pt`) onto the GPU at once (15 x 17 MB).
2. For each patient and each of the 13 DEC-005 conditions: degrade and
   normalise **once**, then run every model on that same volume.
3. Per (model, seed, patient, condition) write one row to
   `results/raw_metrics_v2.csv`: `model, seed, patient_id, grade,
   condition_id, noise, snr, r, thick, dice_wt, dice_tc, dice_et, dice_mean,
   vol_pred_wt_cm3, vol_gt_wt_cm3` (same for tc, et; 1 mm voxels, from the
   NIfTI header spacing), `n_components_wt`, `false_tumor_cm3` (predicted WT
   on slices with no true tumor), and `hd95_wt/tc/et` **only for three
   conditions** - clean, `kspace_snr5_r1.0_t1`, `kspace_snr3_r0.5_t5` -
   because HD95 is slow (empty for the other ten). Grade comes from
   `name_mapping.csv`.
4. Save NIfTI predictions for **seed 0 only**, at those same three
   conditions, for Joshua's MATLAB pipeline, in exactly this format:
   `predictions/<model>_<patient_id>_<condition_id>.nii.gz`, uint8, labels
   0 / 1 = NCR / 2 = edema / 3 = ET, padded back from the 192x192 crop to the
   full 240x240x155 grid with the original affine
   (`python/data/dataset.py::reconstruct_3d_volume` and the crop coordinates
   in `python/data/preprocess.py`); plus `predictions/ground_truth/GT_<id>.nii.gz`.
   Label rule: WT -> 2, TC -> 1, ET -> 3 (TC over WT, ET over TC).
5. Resumable: append rows and skip (patient, condition) pairs already in the CSV.

Acceptance checks (on validation patients, seed-0 checkpoints):
- M0/M1/M4 clean and worst 3D Dice reproduce the training logs' best-epoch
  validation values (e.g. M0 clean ~0.84, M1 worst ~0.75) within ~0.005.
- HD95 both-empty -> 0 on an LGG patient without ET.
- A written NIfTI reloads with the right shape, labels and affine; its WT
  voxel count equals the in-memory prediction's.
- Report time per (patient, condition) so the full test run can be planned:
  74 x 13 = 962 degraded volumes x 15 models.

## Task 2: statistics to DEC-006 (`python/analysis/stats.py`)

Your pipeline's hypothesis block must change to the pre-registered DEC-006:

| | Comparison | Condition | Test |
|---|---|---|---|
| **H1 (primary)** | M4 > M1 | worst (SNR 3, r 0.5, 5 mm) | paired Wilcoxon on seed-averaged per-patient mean 3D Dice |
| **H2** | M4's clean-to-worst Dice drop < M1's | clean vs worst | paired Wilcoxon on per-patient drops |
| **H3** | M4 > M6 (two streams vs early fusion) | worst | paired Wilcoxon |

Holm correction over the three, alpha 0.05. A difference smaller than twice
the seed-to-seed SD is reported as "within training noise" regardless of p.
Secondary (reported, not part of the verdict): M4 vs M0, M5 vs M4, and the
full 13-condition Dice curves per model with 95 % bootstrap CIs (seed 42).
Input: `results/raw_metrics_v2.csv` (Task 1). Update your fake-data generator
to the new columns and models, plant known effects (including a null M4 =
M1, which is what seed 0 suggests), and show the tests recover them.

## Task 3 (if time): literature

Still open from your first brief (`docs/handoff/TASK_NAZIR.md`, Task 3):
verify the references in `docs/PAPER_DRAFT.md`, write Related Work.

## Deliverables

- [ ] `evaluation/evaluate_v2.py` + notebook, checks above passed on validation
- [ ] full test run once all 15 checkpoints exist -> `results/raw_metrics_v2.csv`
      (commit; it is small) + `predictions/` folder shared with Joshua (not in git)
- [ ] `stats.py` on DEC-006 + tests on fake data
- [ ] verdict tables + plain-language `docs/RESULTS_STATS.md` from the real CSV
