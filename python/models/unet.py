"""
Single-Stream 2D U-Net Architecture.
Uses InstanceNorm2d, LeakyReLU(0.2), 5 resolution levels, and skip connections.
Outputs 3 sigmoid channels (WT, TC, ET).
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetSingle(nn.Module):
    """
    Standard Single-Stream 5-level U-Net.
    Used for M0, M1, M2, M3, M5, M6 with adjusted base channels for parameter parity.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 24
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        c = [base_channels * (2**i) for i in range(5)] # e.g. [24, 48, 96, 192, 384]

        # Encoder
        self.enc0 = ConvBlock(in_channels, c[0])
        self.pool0 = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(c[0], c[1])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c[1], c[2])
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c[2], c[3])
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(c[3], c[4])

        # Decoder
        self.up3 = nn.ConvTranspose2d(c[4], c[3], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c[3] * 2, c[3])

        self.up2 = nn.ConvTranspose2d(c[3], c[2], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c[2] * 2, c[2])

        self.up1 = nn.ConvTranspose2d(c[2], c[1], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c[1] * 2, c[1])

        self.up0 = nn.ConvTranspose2d(c[1], c[0], kernel_size=2, stride=2)
        self.dec0 = ConvBlock(c[0] * 2, c[0])

        # Output head
        self.head = nn.Conv2d(c[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)
        p0 = self.pool0(e0)

        e1 = self.enc1(p0)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))

        logits = self.head(d0)
        return torch.sigmoid(logits)
