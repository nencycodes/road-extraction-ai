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

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        self.model.to(device)

    def train_one_epoch(
        self,
        epoch
    ):

        self.model.train()

        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}"
        )

        for images, masks in progress_bar:

            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                masks
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            running_dice += dice_score(
                outputs,
                masks
            )

            running_iou += iou_score(
                outputs,
                masks
            )

            progress_bar.set_postfix(
                Loss=f"{loss.item():.4f}"
            )

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
        print(f"Epoch {epoch}")
        print("=" * 60)

        print(f"Loss : {epoch_loss:.4f}")
        print(f"Dice : {epoch_dice:.4f}")
        print(f"IoU  : {epoch_iou:.4f}")

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

    def fit(
        self,
        epochs
    ):

        for epoch in range(
            1,
            epochs + 1
        ):

            self.train_one_epoch(epoch)