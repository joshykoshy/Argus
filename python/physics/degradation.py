"""
Physics-based Low-Field MRI Degradation Engine in PyTorch.
Implements pseudo k-space truncation (resolution loss) and complex Gaussian / Rician noise.
Deterministic seeding for validation/testing, batched GPU/CPU execution.
"""

import hashlib
import struct
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft

# 13 Degradation Conditions
DEGRADATION_CONDITIONS = [
    {"id": "clean", "snr": None, "r": 1.0},
    {"id": "snr30_r1.0", "snr": 30.0, "r": 1.0},
    {"id": "snr30_r0.75", "snr": 30.0, "r": 0.75},
    {"id": "snr30_r0.5", "snr": 30.0, "r": 0.5},
    {"id": "snr20_r1.0", "snr": 20.0, "r": 1.0},
    {"id": "snr20_r0.75", "snr": 20.0, "r": 0.75},
    {"id": "snr20_r0.5", "snr": 20.0, "r": 0.5},
    {"id": "snr12_r1.0", "snr": 12.0, "r": 1.0},
    {"id": "snr12_r0.75", "snr": 12.0, "r": 0.75},
    {"id": "snr12_r0.5", "snr": 12.0, "r": 0.5},
    {"id": "snr8_r1.0", "snr": 8.0, "r": 1.0},
    {"id": "snr8_r0.75", "snr": 8.0, "r": 0.75},
    {"id": "snr8_r0.5", "snr": 8.0, "r": 0.5},
]

CONDITION_MAP = {c["id"]: c for c in DEGRADATION_CONDITIONS}


def get_deterministic_seed(patient_id: str, slice_idx: int, channel: int, condition_id: str) -> int:
    """Generates an immutable 32-bit integer seed from a unique tuple."""
    key = f"{patient_id}_{slice_idx}_{channel}_{condition_id}".encode("utf-8")
    hash_bytes = hashlib.sha256(key).digest()[:4]
    return struct.unpack(">I", hash_bytes)[0] % (2**31 - 1)


def generate_kspace_mask(shape: Tuple[int, int], r: float, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generates symmetric rectangular k-space truncation mask strictly preserving Hermitian symmetry.
    """
    H, W = shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    if r >= 0.999:
        mask.fill_(1.0)
        return mask

    # Radius of k-space preservation
    kh_rad = int(np.round(H * r / 2.0))
    kw_rad = int(np.round(W * r / 2.0))
    
    ch, cw = H // 2, W // 2
    r_start = max(0, ch - kh_rad)
    r_end = min(H, ch + kh_rad + 1)
    c_start = max(0, cw - kw_rad)
    c_end = min(W, cw + kw_rad + 1)

    mask[r_start:r_end, c_start:c_end] = 1.0
    return mask


def apply_degradation_single_slice(
    x: torch.Tensor,
    mu_brain: torch.Tensor,
    snr: Optional[float],
    r: float,
    patient_id: Optional[str] = None,
    slice_idx: Optional[int] = None,
    condition_id: str = "custom",
    shared_noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Degrades a 4-channel 2D slice (4, H, W).
    1. Pseudo k-space: K = fftshift(fft2(x))
    2. Resolution truncation: K_trunc = K * M(r)
    3. Inverse FFT: x_r = real(ifft2(ifftshift(K_trunc)))
    4. Rician Noise: x_deg = sqrt((x_r + n_real)^2 + (n_imag)^2)
       where sigma = mu_brain / SNR
    """
    C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    # Resolution truncation via pseudo k-space
    if r < 0.999:
        mask = generate_kspace_mask((H, W), r, device=device)
        # fft2 over spatial dimensions [-2, -1]
        K = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        K_trunc = K * mask.unsqueeze(0)
        x_r = torch.real(fft.ifft2(fft.ifftshift(K_trunc, dim=(-2, -1)), dim=(-2, -1)))
    else:
        x_r = x.clone()

    # Rician noise addition
    if snr is None or snr <= 0:
        return x_r

    x_deg = torch.zeros_like(x_r)
    for c in range(C):
        sigma = float(mu_brain[c].item()) / float(snr)
        
        if shared_noise is not None:
            n_real = shared_noise[c, 0] * sigma
            n_imag = shared_noise[c, 1] * sigma
        elif patient_id is not None and slice_idx is not None:
            # Deterministic noise generation
            seed = get_deterministic_seed(patient_id, slice_idx, c, condition_id)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            n_real = torch.randn((H, W), generator=gen, device=device, dtype=dtype) * sigma
            n_imag = torch.randn((H, W), generator=gen, device=device, dtype=dtype) * sigma
        else:
            # Stochastic training noise
            n_real = torch.randn((H, W), device=device, dtype=dtype) * sigma
            n_imag = torch.randn((H, W), device=device, dtype=dtype) * sigma

        # Magnitude of complex noisy signal -> Rician distribution
        x_deg[c] = torch.sqrt((x_r[c] + n_real)**2 + (n_imag)**2)

    return x_deg


class LowFieldDegradation(nn.Module):
    """
    PyTorch Module wrapper for batch/slice degradation.
    """
    def __init__(self):
        super().__init__()
        self.conditions = DEGRADATION_CONDITIONS

    def forward(
        self,
        x: torch.Tensor,
        mu_brain: torch.Tensor,
        condition_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        slice_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, str]:
        if condition_id is None:
            # Uniform random sampling from all 13 conditions during training
            idx = torch.randint(0, len(self.conditions), (1,)).item()
            cond = self.conditions[idx]
        else:
            cond = CONDITION_MAP[condition_id]

        snr = cond["snr"]
        r = cond["r"]
        c_id = cond["id"]

        if x.ndim == 3: # (C, H, W)
            x_deg = apply_degradation_single_slice(
                x, mu_brain, snr, r, patient_id=patient_id, slice_idx=slice_idx, condition_id=c_id
            )
        elif x.ndim == 4: # (B, C, H, W)
            B = x.shape[0]
            x_deg_list = []
            for b in range(B):
                mu_b = mu_brain[b] if mu_brain.ndim == 2 else mu_brain
                deg_b = apply_degradation_single_slice(
                    x[b], mu_b, snr, r, patient_id=patient_id, slice_idx=slice_idx, condition_id=c_id
                )
                x_deg_list.append(deg_b)
            x_deg = torch.stack(x_deg_list, dim=0)
        else:
            raise ValueError(f"Unsupported tensor dimension {x.ndim}")

        return x_deg, c_id
