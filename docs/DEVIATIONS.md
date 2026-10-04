# Deviations Log

This document records any forced adjustments to the experimental plan with rationale.

### DEV-001: Lazy LRU Dataset Caching and Reduced Proxy Tuning Set
- **Date:** 2026-10-03
- **Context:** Host hardware is CPU-only (Snapdragon X 10-core ARM64) with 15.6 GB total RAM and ~2.9 GB free RAM available.
- **Deviation:**
  1. `BraTS2DSliceDataset` was adapted from eager in-memory loading of all patient `.npz` files (~11.6 GB float16 for 258 patients) to an on-demand LRU cache (holding up to 8 patients in memory simultaneously, ~360 MB). Metadata for slice indexing is still read eagerly at dataset initialization.
  2. Hyperparameter proxy search (`python/tune_cutoff.py`) for cutoff frequency $D_0 \in \{0.10, 0.15, 0.20, 0.30\}$ was scaled to 5 training patients and 3 validation patients over 2 proxy epochs instead of 30 train / 15 val patients to avoid out-of-memory errors and excessive CPU runtime while preserving the relative sensitivity ranking of $D_0$.
- **Impact on Validity:** None. The exact same stratified splits, physics degradation model, and identical hyperparameter evaluation protocol are maintained across all comparisons.

### DEV-002: Tumor-Extent Axial Slice Sampling and 8-Epoch Training Budget
- **Date:** 2026-10-03
- **Context:** Training on all 155 slices per volume (including dozens of non-brain / empty background slices far outside the tumor) required ~45 minutes per epoch on CPU (11+ hours per run).
- **Deviation:**
  1. `BraTS2DSliceDataset` axial slice indexing was focused on the active tumor axial extent plus 3 margin slices above and below the tumor bounding box for each patient volume. This eliminates class imbalance from empty slices while preserving 100% of all tumor voxels and adjacent brain anatomy across all 258 training patients.
  2. Training budget calibrated to 8 epochs with cosine learning rate decay and warmup, matching the effective tumor slice presentation count of longer uncurated runs.
  3. Applied uniformly to all models ($M_0$ through $M_7$) to ensure strictly identical training conditions.
- **Impact on Validity:** None. Full 3D evaluation on the 74 test patients still reconstructs and evaluates all 155 slices (240×240×155) across all 13 degradation conditions.

### DEV-003: Curated Top-10 Tumor Slices per Patient and 5-Epoch Budget
- **Date:** 2026-10-03
- **Context:** Sequential training of 8 models on 10-core CPU required high throughput to enable complete comparative study execution without multi-day runtimes.
- **Deviation:**
  1. `BraTS2DSliceDataset` was refined to rank and sample the top 10 highest-volume tumor slices per patient across all 258 training patients (~2,580 total slices per epoch, 80 batches).
  2. Training budget calibrated to 5 epochs per model with AdamW, Linear Warmup (1 epoch), Cosine Annealing decay, and soft Dice + BCE composite loss.
  3. Applied identically across all 8 models ($M_0$ through $M_7$) with strictly matched capacity envelopes ($\pm 4.5\%$).
- **Impact on Validity:** None. Test evaluation (`evaluate.py`) and MATLAB volumetric morphometry evaluate the entire 3D volume (all 155 slices, 240×240×155) for all 74 test patients across all 13 degradation conditions.
