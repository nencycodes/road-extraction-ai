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
    # PATHS
    # ==========================================================

    dataset_root = (
        "/content/drive/MyDrive/"
        "RoadVision-AI/dataset"
    )

    metadata_path = (
        "/content/drive/MyDrive/"
        "RoadVision-AI/dataset/metadata.csv"
    )

    model_save_path = (
        "/content/drive/MyDrive/"
        "RoadVision-AI/models/best_model.pth"
    )

    # ==========================================================
    # DEVICE
    # ==========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Road Extraction AI")
    print("=" * 60)

    print(f"Device: {device}")

    # ==========================================================
    # IMAGE TRANSFORM
    # ==========================================================

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # ==========================================================
    # MASK TRANSFORM
    # ==========================================================

    mask_transform = transforms.Compose([
        transforms.Resize(
            (256, 256),
            interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.ToTensor()
    ])

    # ==========================================================
    # DATASET
    # ==========================================================

    train_dataset = RoadDataset(
        metadata_path=metadata_path,
        dataset_root=dataset_root,
        split="train",
        image_transform=image_transform,
        mask_transform=mask_transform
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

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Batches per epoch: {len(train_loader)}"
    )

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
        device=device,
        save_path=model_save_path
    )

    # ==========================================================
    # TRAIN
    # ==========================================================

    trainer.fit(
        epochs=20
    )


if __name__ == "__main__":
    main()