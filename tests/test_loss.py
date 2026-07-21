"""
test_loss.py

Tests the loss function.
"""

from src.training.loss import get_loss


def test_loss():

    print("=" * 50)
    print("Testing Loss Function")
    print("=" * 50)

    loss = get_loss()

    print(loss)

    print("\n✅ Loss Function Created Successfully")


if __name__ == "__main__":
    test_loss()