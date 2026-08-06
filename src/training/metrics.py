"""
metrics.py

Evaluation metrics for road segmentation.
"""

import torch


def dice_score(predictions, targets, smooth=1e-6):
    """
    Calculates Dice Score.

    Args:
        predictions: Raw model outputs (logits)
        targets: Ground truth masks

    Returns:
        Dice Score (float)
    """

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()

    intersection = (predictions * targets).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum() + targets.sum() + smooth
    )

    return dice.item()


def iou_score(predictions, targets, smooth=1e-6):
    """
    Calculates Intersection over Union (IoU).

    Args:
        predictions: Raw model outputs (logits)
        targets: Ground truth masks

    Returns:
        IoU Score (float)
    """

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()

    intersection = (predictions * targets).float().sum()

    union = (
        predictions.sum()
        + targets.sum()
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.item()