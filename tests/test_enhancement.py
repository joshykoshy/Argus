"""
Tests for Spatial Enhancement parity and correlation between MATLAB and Python.
Verifies CLAHE (adapthisteq vs skimage equalize_adapthist) and top-hat filtering (imtophat vs white_tophat).
Asserts correlation > 0.95.
"""

import subprocess
from pathlib import Path
import numpy as np
import scipy.io as sio
from skimage.exposure import equalize_adapthist
from skimage.morphology import white_tophat, disk
import pytest


def test_enhancement_correlation_matlab_python():
    matlab_dir = Path("C:/Users/Mayan/.gemini/antigravity/scratch/ecte408/matlab/reference")
    in_mat = matlab_dir / "enhancement_test_in.mat"
    out_mat = matlab_dir / "enhancement_test_out.mat"

    # Create synthetic test slice normalized [0, 1]
    np.random.seed(42)
    img = np.random.uniform(0.1, 0.9, size=(192, 192)).astype(np.float32)
    sio.savemat(str(in_mat), {"img": img})

    in_mat_str = str(in_mat).replace('\\', '/')
    out_mat_str = str(out_mat).replace('\\', '/')

    matlab_script = f"""
    data = load('{in_mat_str}');
    img = single(data.img);
    
    % CLAHE
    img_clahe = adapthisteq(img, 'ClipLimit', 0.01, 'NumTiles', [8, 8]);
    
    % Top-hat with disk radius 3
    se = strel('disk', 3);
    img_tophat = imtophat(img, se);
    
    save('{out_mat_str}', 'img_clahe', 'img_tophat');
    exit;
    """

    cmd = ["matlab", "-batch", matlab_script]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"MATLAB run failed: {res.stderr}"

    mat_out = sio.loadmat(str(out_mat))
    mat_clahe = mat_out["img_clahe"]
    mat_tophat = mat_out["img_tophat"]

    # Python CLAHE & Top-hat
    py_clahe = equalize_adapthist(img, clip_limit=0.01, nbins=256)
    py_tophat = white_tophat(img, disk(3))

    # Pearson correlation
    corr_clahe = np.corrcoef(py_clahe.flatten(), mat_clahe.flatten())[0, 1]
    corr_tophat = np.corrcoef(py_tophat.flatten(), mat_tophat.flatten())[0, 1]

    assert corr_clahe > 0.95, f"CLAHE correlation {corr_clahe:.4f} < 0.95"
    assert corr_tophat > 0.95, f"Top-hat correlation {corr_tophat:.4f} < 0.95"
    print(f"CLAHE correlation: {corr_clahe:.4f}, Top-hat correlation: {corr_tophat:.4f}")
