"""
test_optimizer.py

Tests the optimizer.
"""

from src.models.unet import UNet
from src.training.optimizer import get_optimizer


def test_optimizer():

    model = UNet()

    optimizer = get_optimizer(model)

    print("=" * 50)
    print("Testing Optimizer")
    print("=" * 50)

    print(optimizer)

    print("\n✅ Optimizer Created Successfully")


if __name__ == "__main__":
    test_optimizer()