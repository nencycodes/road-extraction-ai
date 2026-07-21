"""
decoder.py

Contains the Decoder Block used in the expanding path
of the U-Net architecture.
"""

import torch
import torch.nn as nn

from src.models.double_conv import DoubleConv


class DecoderBlock(nn.Module):
    """
    Decoder Block

    Performs:

    Input Feature Map
            │
            ▼
    Transposed Convolution (Upsampling)
            │
            ▼
    Concatenate Skip Connection
            │
            ▼
        Double Convolution
            │
            ▼
        Refined Feature Map
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            dropout=dropout
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x