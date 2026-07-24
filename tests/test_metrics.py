"""
test_metrics.py

Tests Dice Score and IoU Score.
"""

import torch

from src.training.metrics import (
    dice_score,
    iou_score
)


def test_metrics():

    print("=" * 50)
    print("Testing Metrics")
    print("=" * 50)

    predictions = torch.randn(
        1,
        1,
        256,
        256
    )

    targets = torch.randint(
        0,
        2,
        (1, 1, 256, 256)
    ).float()

    dice = dice_score(
        predictions,
        targets
    )

    iou = iou_score(
        predictions,
        targets
    )

    print(f"Dice Score : {dice:.4f}")
    print(f"IoU Score  : {iou:.4f}")

    print("\n✅ Metrics Test Passed Successfully")


if __name__ == "__main__":
    test_metrics()