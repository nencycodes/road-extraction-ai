"""
trainer.py

Handles the complete training pipeline.
"""

import os
import torch
from tqdm import tqdm

from src.training.metrics import (
    dice_score,
    iou_score
)


class Trainer:
    """
    Trainer class responsible for training the U-Net model.
    """

    def __init__(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        save_path="models/best_model.pth"
    ):

        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path

        self.best_loss = float("inf")

        # Create models directory if it doesn't exist
        os.makedirs(
            os.path.dirname(self.save_path),
            exist_ok=True
        )

        # Move model to GPU/CPU
        self.model.to(self.device)

    def train_one_epoch(self, epoch):

        self.model.train()

        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}"
        )

        for images, masks in progress_bar:

            # Move batch to GPU/CPU
            images = images.to(
                self.device,
                non_blocking=True
            )

            masks = masks.to(
                self.device,
                non_blocking=True
            )

            # BCEWithLogitsLoss expects float targets
            masks = masks.float()

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward Pass
            outputs = self.model(images)

            # Calculate Loss
            loss = self.criterion(
                outputs,
                masks
            )

            # Backpropagation
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Metrics
            running_loss += loss.item()

            current_dice = dice_score(
                outputs,
                masks
            )

            current_iou = iou_score(
                outputs,
                masks
            )

            running_dice += current_dice
            running_iou += current_iou

            # Update Progress Bar
            progress_bar.set_postfix(
                Loss=f"{loss.item():.4f}",
                Dice=f"{current_dice:.4f}",
                IoU=f"{current_iou:.4f}"
            )

        # Average Metrics
        epoch_loss = (
            running_loss /
            len(self.train_loader)
        )

        epoch_dice = (
            running_dice /
            len(self.train_loader)
        )

        epoch_iou = (
            running_iou /
            len(self.train_loader)
        )

        print("\n" + "=" * 60)
        print(f"Epoch {epoch} Summary")
        print("=" * 60)
        print(f"Loss : {epoch_loss:.4f}")
        print(f"Dice : {epoch_dice:.4f}")
        print(f"IoU  : {epoch_iou:.4f}")

        # Save Best Model
        if epoch_loss < self.best_loss:

            self.best_loss = epoch_loss

            torch.save(
                self.model.state_dict(),
                self.save_path
            )

            print("\n✅ Best Model Saved!")

        return (
            epoch_loss,
            epoch_dice,
            epoch_iou
        )

    def fit(self, epochs):

        print("=" * 60)
        print("🚀 Starting Training")
        print("=" * 60)

        for epoch in range(
            1,
            epochs + 1
        ):

            self.train_one_epoch(epoch)

        print("\n" + "=" * 60)
        print("Training Completed Successfully!")
        print("=" * 60)