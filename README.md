# ECTE408 Low-Field MRI Brain Tumor Segmentation Study

> **Title:** Does Frequency-Band Decomposition Improve Brain Tumor Segmentation Robustness Under Simulated Low-Field MRI Degradation?

## Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure BraTS 2020 dataset is available at data/cache/ (pre-processed NPZ files)

# 3. Train all models
python -m python.train_all

# 4. Evaluate on test set
python -m python.evaluate

# 5. Run MATLAB analysis (requires MATLAB R2025a + Image Processing Toolbox)
matlab -batch "addpath('matlab'); addpath('matlab/reference'); spectral_analysis; enhancement_demo; volumetrics; diagnostic_report; exit"

# 6. Statistical analysis & figures
python -m python.analysis.stats
python -m python.analysis.plots
```

## Project Structure
```
ecte408/
├── configs/              # base.yaml with all hyperparameters
├── data/
│   ├── cache/            # Preprocessed patient .npz files (369 patients)
│   ├── splits/           # split_v1.json (70/10/20 stratified split, seed=42)
│   └── manifest.csv      # Patient manifest with paths and HGG/LGG grade
├── docs/                 # DECISIONS.md, DEVIATIONS.md, ENVIRONMENT.md
├── python/
│   ├── data/             # dataset.py, preprocess.py, splits.py
│   ├── physics/          # degradation.py, decomposition.py
│   ├── models/           # unet.py, dual_stream.py, factory.py
│   ├── metrics/          # evaluation.py (Dice, HD95, Volume)
│   ├── analysis/         # stats.py, plots.py, m0_curve.py
│   ├── train.py          # Single model training loop
│   ├── train_all.py      # Sequential training of all models
│   └── evaluate.py       # Full 3D test-set evaluation
├── matlab/
│   ├── reference/        # degrade_slice.m, decompose_slice.m (parity-verified)
│   ├── spectral_analysis.m
│   ├── enhancement_demo.m
│   ├── volumetrics.m     # regionprops3 biomarkers (MATLAB)
│   └── diagnostic_report.m
├── results/              # Checkpoints, logs, raw_metrics.csv
├── figures/              # All publication figures
└── tables/               # model_complexity.csv, headline_metrics.csv
```

## Dataset
**BraTS 2020** — 369 patients, 4 MRI sequences (T1, T1ce, T2, FLAIR), 240×240×155 @ 1mm isotropic.
Labels: 1=NCR/NET, 2=Edema, 4=Enhancing Tumor. Internally remapped: 4→3.

## Models
| ID | Description | Params |
|----|-------------|--------|
| M0 | Clean-trained U-Net (baseline) | 4.37M |
| M1 | U-Net + degradation augmentation | 4.37M |
| M2 | NLM denoiser + U-Net | 4.37M |
| M3 | Low-pass + CLAHE + U-Net | 4.37M |
| M4 | **Proposed Dual-Stream U-Net** (D0=0.20) | 4.17M |
| M5 | Low-band only ablation | 4.37M |
| M6 | Early-fusion 8-channel U-Net | 4.37M |
| M7 | High-band enhanced dual-stream | 4.17M |
