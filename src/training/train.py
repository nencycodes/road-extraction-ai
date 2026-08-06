"""
train.py

Main training script for Road Extraction AI.
"""

import torch

from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset.road_dataset import RoadDataset
from src.models.unet import UNet

from src.training.loss import get_loss
from src.training.optimizer import get_optimizer
from src.training.trainer import Trainer


def main():

    # ==========================================================
    # DATASET PATHS
    # ==========================================================

    train_image_dir = "/content/drive/MyDrive/RoadVision-AI/dataset/tiff/train"

    train_mask_dir = "/content/drive/MyDrive/RoadVision-AI/dataset/tiff/train_labels"

    # ==========================================================
    # TRANSFORMS
    # ==========================================================

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # ==========================================================
    # DATASET
    # ==========================================================

    train_dataset = RoadDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=transform
    )

    # ==========================================================
    # DATALOADER
    # ==========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # ==========================================================
    # DEVICE
    # ==========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Device :", device)
    print("=" * 60)

    # ==========================================================
    # MODEL
    # ==========================================================

    model = UNet()

    # ==========================================================
    # LOSS
    # ==========================================================

    criterion = get_loss()

    # ==========================================================
    # OPTIMIZER
    # ==========================================================

    optimizer = get_optimizer(
        model,
        learning_rate=1e-3
    )

    # ==========================================================
    # TRAINER
    # ==========================================================

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    # ==========================================================
    # START TRAINING
    # ==========================================================

    trainer.fit(
        epochs=20
    )


if __name__ == "__main__":
    main()