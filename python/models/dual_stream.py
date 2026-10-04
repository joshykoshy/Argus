"""
Dual-Stream Frequency-Decomposition U-Net Architecture (Proposed M4 / M7).
Features two parallel encoder streams:
- Stream A: Low-frequency band representation
- Stream B: High-frequency band representation
Scale-wise fusion via 1x1 convolutions feeding into a shared multi-scale decoder.
"""

from typing import Tuple
import torch
import torch.nn as nn
from python.models.unet import ConvBlock


class DualStreamUNet(nn.Module):
    """
    Dual-stream architecture fusing low and high frequency representations at each encoder scale.
    Base width adjusted to 17 channels to precisely match single U-Net capacity within +-3%.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 17
    ):
        super().__init__()
        self.base_channels = base_channels
        c = [int(base_channels * (2**i)) for i in range(5)] # [17, 34, 68, 136, 272]

        # Encoder Stream A (Low Frequencies)
        self.encA0 = ConvBlock(in_channels, c[0])
        self.poolA0 = nn.MaxPool2d(2)
        self.encA1 = ConvBlock(c[0], c[1])
        self.poolA1 = nn.MaxPool2d(2)
        self.encA2 = ConvBlock(c[1], c[2])
        self.poolA2 = nn.MaxPool2d(2)
        self.encA3 = ConvBlock(c[2], c[3])
        self.poolA3 = nn.MaxPool2d(2)
        self.bottleneckA = ConvBlock(c[3], c[4])

        # Encoder Stream B (High Frequencies)
        self.encB0 = ConvBlock(in_channels, c[0])
        self.poolB0 = nn.MaxPool2d(2)
        self.encB1 = ConvBlock(c[0], c[1])
        self.poolB1 = nn.MaxPool2d(2)
        self.encB2 = ConvBlock(c[1], c[2])
        self.poolB2 = nn.MaxPool2d(2)
        self.encB3 = ConvBlock(c[2], c[3])
        self.poolB3 = nn.MaxPool2d(2)
        self.bottleneckB = ConvBlock(c[3], c[4])

        # Scale-wise Fusion Layers (1x1 Conv fusing Concat(StreamA, StreamB))
        self.fuse0 = nn.Conv2d(c[0] * 2, c[0], kernel_size=1)
        self.fuse1 = nn.Conv2d(c[1] * 2, c[1], kernel_size=1)
        self.fuse2 = nn.Conv2d(c[2] * 2, c[2], kernel_size=1)
        self.fuse3 = nn.Conv2d(c[3] * 2, c[3], kernel_size=1)
        self.fuse_b = nn.Conv2d(c[4] * 2, c[4], kernel_size=1)

        # Shared Multi-Scale Decoder
        self.up3 = nn.ConvTranspose2d(c[4], c[3], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c[3] * 2, c[3])

        self.up2 = nn.ConvTranspose2d(c[3], c[2], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c[2] * 2, c[2])

        self.up1 = nn.ConvTranspose2d(c[2], c[1], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c[1] * 2, c[1])

        self.up0 = nn.ConvTranspose2d(c[1], c[0], kernel_size=2, stride=2)
        self.dec0 = ConvBlock(c[0] * 2, c[0])

        # Output Head
        self.head = nn.Conv2d(c[0], out_channels, kernel_size=1)

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        # Stream A (Low)
        eA0 = self.encA0(x_low)
        pA0 = self.poolA0(eA0)
        eA1 = self.encA1(pA0)
        pA1 = self.poolA1(eA1)
        eA2 = self.encA2(pA1)
        pA2 = self.poolA2(eA2)
        eA3 = self.encA3(pA2)
        pA3 = self.poolA3(eA3)
        bA = self.bottleneckA(pA3)

        # Stream B (High)
        eB0 = self.encB0(x_high)
        pB0 = self.poolB0(eB0)
        eB1 = self.encB1(pB0)
        pB1 = self.poolB1(eB1)
        eB2 = self.encB2(pB1)
        pB2 = self.poolB2(eB2)
        eB3 = self.encB3(pB2)
        pB3 = self.poolB3(eB3)
        bB = self.bottleneckB(pB3)

        # Scale-wise Feature Fusion
        f0 = self.fuse0(torch.cat([eA0, eB0], dim=1))
        f1 = self.fuse1(torch.cat([eA1, eB1], dim=1))
        f2 = self.fuse2(torch.cat([eA2, eB2], dim=1))
        f3 = self.fuse3(torch.cat([eA3, eB3], dim=1))
        fb = self.fuse_b(torch.cat([bA, bB], dim=1))

        # Shared Decoder
        d3 = self.dec3(torch.cat([self.up3(fb), f3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), f2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), f1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), f0], dim=1))

        logits = self.head(d0)
        return torch.sigmoid(logits)
