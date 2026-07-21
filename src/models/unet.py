"""
unet.py

Complete Improved U-Net Architecture.
"""

import torch
import torch.nn as nn

from src.models.encoder import EncoderBlock
from src.models.decoder import DecoderBlock
from src.models.double_conv import DoubleConv


class UNet(nn.Module):

    def __init__(self, dropout=0.2):
        super().__init__()

        # ---------------- Encoder ---------------- #

        self.encoder1 = EncoderBlock(3, 64, dropout)
        self.encoder2 = EncoderBlock(64, 128, dropout)
        self.encoder3 = EncoderBlock(128, 256, dropout)
        self.encoder4 = EncoderBlock(256, 512, dropout)

        # ---------------- Bottleneck ---------------- #

        self.bottleneck = DoubleConv(
            512,
            1024,
            dropout
        )

        # ---------------- Decoder ---------------- #

        self.decoder4 = DecoderBlock(
            1024,
            512,
            dropout
        )

        self.decoder3 = DecoderBlock(
            512,
            256,
            dropout
        )

        self.decoder2 = DecoderBlock(
            256,
            128,
            dropout
        )

        self.decoder1 = DecoderBlock(
            128,
            64,
            dropout
        )

        # ---------------- Output Layer ---------------- #

        self.output = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # -------- Encoder -------- #

        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        # -------- Bottleneck -------- #

        x = self.bottleneck(x)

        # -------- Decoder -------- #

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.output(x)