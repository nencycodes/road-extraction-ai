"""
double_conv.py

Contains the Double Convolution Block used throughout the U-Net architecture.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers followed by
    Batch Normalization and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)