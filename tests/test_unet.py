"""
test_unet.py

Tests the complete U-Net architecture.
"""

import torch

from src.models.unet import UNet


def test_unet():

    print("=" * 60)
    print("Testing Complete U-Net")
    print("=" * 60)

    model = UNet()

    x = torch.randn(1, 3, 256, 256)

    output = model(x)

    print(f"Input Shape  : {x.shape}")
    print(f"Output Shape : {output.shape}")

    assert output.shape == (1, 1, 256, 256)

    print("\n✅ U-Net Test Passed Successfully!")


if __name__ == "__main__":
    test_unet()