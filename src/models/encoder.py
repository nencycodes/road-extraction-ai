"""
encoder.py

Contains the Encoder Block used in the contracting path
of the U-Net architecture.
"""

import torch
import torch.nn as nn

from src.models.double_conv import DoubleConv


class EncoderBlock(nn.Module):
    """
    Encoder Block

    Performs:

    Input
        │
        ▼
    Double Convolution
        │
        ├────────► Skip Features
        │
        ▼
    Max Pooling
        │
        ▼
    Downsampled Features
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv = DoubleConv(
            in_channels,
            out_channels,
            dropout
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        features = self.conv(x)

        pooled = self.pool(features)

        return features, pooled