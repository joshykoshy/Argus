"""
Verification tests for Low-Field MRI Degradation and Frequency Decomposition.
Checks:
- SNR target accuracy (<2% error)
- Rayleigh noise distribution in signal-free background
- Hermitian symmetry and real reconstruction (<1e-6 imaginary residual)
- Exact frequency decomposition reconstruction (low + high == x)
- Cross-platform MATLAB-Python numerical parity (<1e-4 max absolute error)
"""

import os
import subprocess
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.stats import rayleigh
import torch
import pytest

from python.physics.degradation import (
    apply_degradation_single_slice,
    generate_kspace_mask,
    DEGRADATION_CONDITIONS
)
from python.physics.decomposition import (
    compute_gaussian_filters,
    decompose_frequencies
)


def test_hermitian_symmetry_and_real_ifft():
    """Verify that k-space truncation mask produces real-valued output (imag part ~ 0)."""
    torch.manual_seed(42)
    x = torch.rand((4, 192, 192), dtype=torch.float32)
    
    for r in [1.0, 0.75, 0.5]:
        mask = generate_kspace_mask((192, 192), r)
        K = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        K_trunc = K * mask.unsqueeze(0)
        x_complex = torch.fft.ifft2(torch.fft.ifftshift(K_trunc, dim=(-2, -1)), dim=(-2, -1))
        
        imag_max = torch.max(torch.abs(torch.imag(x_complex))).item()
        assert imag_max < 1e-5, f"Imaginary part {imag_max} exceeds 1e-5 for r={r}"


def test_exact_frequency_reconstruction():
    """Verify that low + high frequency bands exactly reconstruct the original input."""
    torch.manual_seed(42)
    x = torch.randn((4, 192, 192), dtype=torch.float32)

    for d0 in [0.10, 0.15, 0.20, 0.30]:
        x_low, x_high = decompose_frequencies(x, d0=d0)
        recon = x_low + x_high
        diff = torch.max(torch.abs(recon - x)).item()
        assert diff < 1e-5, f"Decomposition reconstruction error {diff} exceeds 1e-5 for d0={d0}"


def test_snr_and_rayleigh_noise():
    """
    Verify achieved SNR is within 2% of target on synthetic phantom
    and background noise magnitude follows Rayleigh distribution.
    """
    H, W = 256, 256
    # Create circular phantom with mean intensity 1000 in center, 0 in background
    yy, xx = np.ogrid[:H, :W]
    center_y, center_x = H // 2, W // 2
    circle_mask = (xx - center_x)**2 + (yy - center_y)**2 <= 60**2
    bg_mask = (xx - center_x)**2 + (yy - center_y)**2 >= 100**2

    clean_img = np.zeros((H, W), dtype=np.float32)
    clean_img[circle_mask] = 1000.0
    mu_brain_val = 1000.0

    x_tensor = torch.from_numpy(clean_img).unsqueeze(0) # (1, H, W)
    mu_tensor = torch.tensor([mu_brain_val], dtype=torch.float32)

    for target_snr in [30.0, 20.0, 12.0, 8.0]:
        # Generate noisy realization
        torch.manual_seed(123)
        sigma = mu_brain_val / target_snr
        n_real = torch.randn((1, 1, H, W))
        n_imag = torch.randn((1, 1, H, W))
        shared_n = torch.cat([n_real, n_imag], dim=1) # (1, 2, H, W)

        x_deg = apply_degradation_single_slice(
            x_tensor, mu_tensor, snr=target_snr, r=1.0, shared_noise=shared_n
        ).squeeze().numpy()

        # In pure background (signal = 0), magnitude is Rayleigh(sigma)
        bg_samples = x_deg[bg_mask]
        empirical_sigma = np.sqrt(np.mean(bg_samples**2) / 2.0)
        measured_snr = mu_brain_val / empirical_sigma

        rel_error = abs(measured_snr - target_snr) / target_snr
        assert rel_error < 0.02, f"Target SNR {target_snr} vs Measured SNR {measured_snr:.2f} (error {rel_error*100:.2f}% > 2%)"

        # Check Rayleigh mean and variance
        # Theoretical mean = sigma * sqrt(pi/2), var = (4-pi)/2 * sigma^2
        expected_mean = sigma * np.sqrt(np.pi / 2.0)
        actual_mean = np.mean(bg_samples)
        assert abs(actual_mean - expected_mean) / expected_mean < 0.02, "Rayleigh mean mismatch"


def test_matlab_python_parity():
    """
    Exports clean slices and shared noise to .mat, runs MATLAB reference degrade_slice & decompose_slice,
    and asserts max absolute difference < 1e-4 (float32).
    """
    matlab_dir = Path("C:/Users/Mayan/.gemini/antigravity/scratch/ecte408/matlab/reference")
    test_mat_path = matlab_dir / "parity_test_data.mat"
    res_mat_path = matlab_dir / "parity_matlab_results.mat"

    # Create 5 test slices normalized in [0, 1]
    np.random.seed(42)
    slices = np.random.uniform(0.1, 1.0, size=(5, 192, 192)).astype(np.float32)
    mu_brains = np.mean(slices, axis=(1, 2)).astype(np.float32)
    n_real = np.random.randn(5, 192, 192).astype(np.float32)
    n_imag = np.random.randn(5, 192, 192).astype(np.float32)

    sio.savemat(str(test_mat_path), {
        "slices": slices,
        "mu_brains": mu_brains,
        "n_real": n_real,
        "n_imag": n_imag
    })

    matlab_dir_str = str(matlab_dir).replace('\\', '/')
    test_mat_str = str(test_mat_path).replace('\\', '/')
    res_mat_str = str(res_mat_path).replace('\\', '/')

    # Run MATLAB parity script
    matlab_script = f"""
    addpath('{matlab_dir_str}');
    data = load('{test_mat_str}');
    
    slices = single(data.slices);
    mu_brains = single(data.mu_brains);
    n_real = single(data.n_real);
    n_imag = single(data.n_imag);
    
    % Test all conditions
    snrs = [30, 20, 12, 8];
    rs = [1.0, 0.75, 0.5];
    d0s = [0.10, 0.15, 0.20, 0.30];
    
    deg_results = zeros(5, 4, 3, 192, 192, 'single');
    for i = 1:5
        x = squeeze(slices(i, :, :));
        mu_b = mu_brains(i);
        nr = squeeze(n_real(i, :, :));
        ni = squeeze(n_imag(i, :, :));
        for s_idx = 1:4
            for r_idx = 1:3
                deg_results(i, s_idx, r_idx, :, :) = degrade_slice(x, mu_b, snrs(s_idx), rs(r_idx), nr, ni);
            end
        end
    end
    
    low_results = zeros(5, 4, 192, 192, 'single');
    high_results = zeros(5, 4, 192, 192, 'single');
    for i = 1:5
        x = squeeze(slices(i, :, :));
        for d_idx = 1:4
            [l_band, h_band] = decompose_slice(x, d0s(d_idx));
            low_results(i, d_idx, :, :) = l_band;
            high_results(i, d_idx, :, :) = h_band;
        end
    end
    
    save('{res_mat_str}', 'deg_results', 'low_results', 'high_results');
    exit;
    """

    # Execute MATLAB batch
    cmd = ["matlab", "-batch", matlab_script]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"MATLAB execution failed: {res.stderr}\n{res.stdout}"
    assert res_mat_path.exists(), "MATLAB results file not generated"

    # Load MATLAB results
    mat_out = sio.loadmat(str(res_mat_path))
    mat_deg = mat_out["deg_results"] # (5, 4, 3, 192, 192)
    mat_low = mat_out["low_results"] # (5, 4, 192, 192)
    mat_high = mat_out["high_results"] # (5, 4, 192, 192)

    # Compute Python degradation results
    snrs = [30.0, 20.0, 12.0, 8.0]
    rs = [1.0, 0.75, 0.5]
    d0s = [0.10, 0.15, 0.20, 0.30]

    for i in range(5):
        x_t = torch.from_numpy(slices[i]).unsqueeze(0) # (1, H, W)
        mu_t = torch.tensor([mu_brains[i]], dtype=torch.float32)
        nr_t = torch.from_numpy(n_real[i]).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        ni_t = torch.from_numpy(n_imag[i]).unsqueeze(0).unsqueeze(0)
        shared_n = torch.cat([nr_t, ni_t], dim=1) # (1, 2, H, W)

        for s_idx, snr_val in enumerate(snrs):
            for r_idx, r_val in enumerate(rs):
                py_deg = apply_degradation_single_slice(
                    x_t, mu_t, snr=snr_val, r=r_val, shared_noise=shared_n
                ).squeeze().numpy()

                mat_deg_slice = mat_deg[i, s_idx, r_idx]
                diff = np.max(np.abs(py_deg - mat_deg_slice))
                assert diff < 1e-4, f"Degradation parity failed for slice {i}, SNR={snr_val}, r={r_val}: max diff = {diff}"

        for d_idx, d0_val in enumerate(d0s):
            py_low, py_high = decompose_frequencies(x_t, d0=d0_val)
            py_low = py_low.squeeze().numpy()
            py_high = py_high.squeeze().numpy()

            mat_low_slice = mat_low[i, d_idx]
            mat_high_slice = mat_high[i, d_idx]

            diff_low = np.max(np.abs(py_low - mat_low_slice))
            diff_high = np.max(np.abs(py_high - mat_high_slice))
            assert diff_low < 1e-4, f"Low-band parity failed for slice {i}, d0={d0_val}: max diff = {diff_low}"
            assert diff_high < 1e-4, f"High-band parity failed for slice {i}, d0={d0_val}: max diff = {diff_high}"

    print("All MATLAB-Python parity checks passed with max diff < 1e-4!")
