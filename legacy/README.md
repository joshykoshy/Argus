# Legacy: the first (v1) pipeline

Kept for transparency; **nothing here is used by the final study or the paper.**

The v1 pipeline trained eight models on the 10 largest-tumour slices per patient, validated on tumour-bearing
slices only, and used a milder degradation range (SNR 30-8, noise added after k-space truncation). On full
volumes those models drew tumour on every tumour-free slice (true 3D Dice about 0.2, finding C11 in
`docs/LAB_NOTEBOOK.md`), so the study was redesigned (DEC-005 to DEC-007 in `docs/DECISIONS.md`).

| Folder | What it was | Replaced by |
|---|---|---|
| `python/` | v1 training (`train.py`, `train_all.py`), evaluation (`evaluate.py`), cutoff tuning, v1 figure scripts | `training/train_v2.py`, `evaluation/evaluate_v2.py` |
| `configs/base.yaml` | v1 hyperparameters | DEC-007 protocol in `training/train_v2.py` |
| `results/M0`-`M7` | v1 checkpoints (seed 0, 5 epochs) | `results/train_v3/` (5 models x 3 seeds) |
| `results/*.csv` | v1 training summary, cutoff search, v1 M0 degradation curve | `results/raw_metrics_v2.csv` |
| `matlab/` | v1 spectral figures (hard-coded claims), enhancement demo (dropped models M3/M7), v1 physics explainer | `matlab/spectral_check.m`, `matlab/diagnostic_report.m` |
| `figures/` | figures of the above | `figures/` |
| `docs/` | v1 spectral findings (one claim did not hold), v1 deviations log | `docs/PAPER_DRAFT.md`, `docs/LAB_NOTEBOOK.md` |
