"""
optimizer.py

Creates and returns the optimizer used for training.
"""

import torch.optim as optim


def get_optimizer(model, learning_rate=1e-3):
    """
    Creates Adam optimizer.

    Args:
        model: Neural network model
        learning_rate: Learning rate

    Returns:
        Adam optimizer
    """

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    return optimizer