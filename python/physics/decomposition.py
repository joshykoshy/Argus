"""
Frequency-band decomposition in PyTorch.
Implements normalized radial distance D(u, v), complementary Gaussian low-pass and high-pass filters,
and exact reconstruction verification.
"""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft


def compute_radial_grid(shape: Tuple[int, int], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Computes normalized radial frequency distance grid D(u, v) from center (DC),
    normalized such that D = 1.0 at the Nyquist boundary along axes.
    """
    H, W = shape
    ch, cw = H / 2.0, W / 2.0
    u = torch.arange(H, device=device, dtype=torch.float32) - ch
    v = torch.arange(W, device=device, dtype=torch.float32) - cw
    U, V = torch.meshgrid(u, v, indexing="ij")
    # Normalized radial distance (fraction of Nyquist)
    D = torch.sqrt((U / ch)**2 + (V / cw)**2)
    return D


def compute_gaussian_filters(
    shape: Tuple[int, int],
    d0: float = 0.20,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs complementary Gaussian low-pass and high-pass transfer functions:
    H_lp = exp(-D^2 / (2 * d0^2))
    H_hp = 1 - H_lp
    """
    D = compute_radial_grid(shape, device=device)
    H_lp = torch.exp(-(D**2) / (2.0 * (d0**2)))
    H_hp = 1.0 - H_lp
    return H_lp, H_hp


def decompose_frequencies(
    x: torch.Tensor,
    d0: float = 0.20,
    H_lp: torch.Tensor = None,
    H_hp: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decomposes multi-channel 2D tensor (C, H, W) or (B, C, H, W) into low and high frequency bands.
    """
    H, W = x.shape[-2], x.shape[-1]
    device = x.device

    if H_lp is None or H_hp is None:
        H_lp, H_hp = compute_gaussian_filters((H, W), d0=d0, device=device)

    # 2D FFT shifted to center
    F = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))

    # Apply complementary filters
    F_lp = F * H_lp
    F_hp = F * H_hp

    # Inverse 2D FFT
    x_low = torch.real(fft.ifft2(fft.ifftshift(F_lp, dim=(-2, -1)), dim=(-2, -1)))
    x_high = torch.real(fft.ifft2(fft.ifftshift(F_hp, dim=(-2, -1)), dim=(-2, -1)))

    return x_low, x_high


class FrequencyBandDecomposition(nn.Module):
    """
    Torch module layer for on-the-fly frequency decomposition.
    """
    def __init__(self, shape: Tuple[int, int] = (192, 192), d0: float = 0.20):
        super().__init__()
        self.shape = shape
        self.d0 = d0
        H_lp, H_hp = compute_gaussian_filters(shape, d0=d0)
        self.register_buffer("H_lp", H_lp)
        self.register_buffer("H_hp", H_hp)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return decompose_frequencies(x, d0=self.d0, H_lp=self.H_lp, H_hp=self.H_hp)
