"""
loss.py

Contains loss functions used for road segmentation.
"""

import torch.nn as nn


def get_loss():
    """
    Returns the loss function used during training.

    We use BCEWithLogitsLoss because road extraction is a
    binary segmentation problem.

    Background -> 0
    Road       -> 1
    """

    return nn.BCEWithLogitsLoss()