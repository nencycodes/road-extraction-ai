"""
test_decoder.py

Tests the Decoder Block.
"""

import torch

from src.models.decoder import DecoderBlock


def test_decoder():

    print("=" * 50)
    print("Testing Decoder Block")
    print("=" * 50)

    decoder = DecoderBlock(
        in_channels=128,
        out_channels=64
    )

    x = torch.randn(1, 128, 128, 128)

    skip = torch.randn(1, 64, 256, 256)

    output = decoder(x, skip)

    print(f"Input Shape   : {x.shape}")
    print(f"Skip Shape    : {skip.shape}")
    print(f"Output Shape  : {output.shape}")

    assert output.shape == (1, 64, 256, 256)

    print("\n✅ Decoder Block Test Passed Successfully!")


if __name__ == "__main__":
    test_decoder()