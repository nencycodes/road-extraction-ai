"""
test_double_conv.py

Tests the DoubleConv block.
"""

import torch

from src.models.double_conv import DoubleConv


def test_double_conv():

    print("=" * 50)
    print("Testing DoubleConv Block")
    print("=" * 50)

    # Create model
    model = DoubleConv(
        in_channels=3,
        out_channels=64
    )

    # Dummy input
    x = torch.randn(1, 3, 256, 256)

    # Forward pass
    output = model(x)

    # Display results
    print(f"Input Shape  : {x.shape}")
    print(f"Output Shape : {output.shape}")

    # Verify output shape
    assert output.shape == (1, 64, 256, 256)

    print("\n✅ DoubleConv Block Test Passed Successfully!")


if __name__ == "__main__":
    test_double_conv()