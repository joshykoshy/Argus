# Environment and Prerequisites Report (Gate G0)

## System Specification
- **Operating System:** Windows 11 (Architecture: ARM64)
- **CPU:** Snapdragon(R) X 10-core X1P64100 @ 3.40 GHz (10 logical processors)
- **RAM / Storage:** 179.80 GB free disk space on `C:\`
- **GPU:** Qualcomm(R) Adreno(TM) X1-85 GPU (PyTorch device: CPU optimized with vectorization)
- **Python Version:** 3.11.9
- **MATLAB Version:** 25.1.0.2943329 (R2025a)
  - **Toolboxes Verified:**
    - Image Processing Toolbox (Verified)
    - Statistics and Machine Learning Toolbox (Verified)
    - Signal Processing Toolbox (Verified)
    - Symbolic Math Toolbox (Verified)
    - Control System Toolbox (Verified)

## Python Environment
- **Torch:** 2.9.1+cpu
- **Key Packages:** `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`, `pytest`, `nibabel`, `scikit-image`, `tensorboard`, `medpy`, `SimpleITK`, `kaggle`

## Dataset Status
- **Target Dataset:** Kaggle `awsaf49/brats20-dataset-training-validation` (BraTS 2020)
- **Verified Local Path:** `C:\Users\Mayan\.cache\kagglehub\datasets\awsaf49\brats20-dataset-training-validation\versions\1\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData`
- **Training Cases:** 369 directories confirmed present + `name_mapping.csv` & `survival_info.csv`.
- **Gate G0 Status:** PASSED.
