"""
road_dataset.py

Custom PyTorch Dataset for the Massachusetts Roads Dataset.
"""

import os

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    """
    Dataset for binary road segmentation.

    Uses metadata.csv as the source of truth for
    image-mask relationships.
    """

    def __init__(
        self,
        metadata_path,
        dataset_root,
        split="train",
        image_transform=None,
        mask_transform=None
    ):

        self.dataset_root = dataset_root
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Read metadata
        metadata = pd.read_csv(metadata_path)

        # Keep only requested split
        metadata = metadata[
            metadata["split"] == split
        ].copy()

        # Keep only files that actually exist
        valid_rows = []

        for _, row in metadata.iterrows():

            image_path = os.path.join(
                dataset_root,
                row["tiff_image_path"]
            )

            mask_path = os.path.join(
                dataset_root,
                row["tif_label_path"]
            )

            if (
                os.path.isfile(image_path)
                and
                os.path.isfile(mask_path)
            ):

                valid_rows.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path
                    }
                )

        self.samples = valid_rows

        print(
            f"{split.upper()} dataset: "
            f"{len(self.samples)} valid image-mask pairs."
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # Load image
        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        # Load mask
        mask = Image.open(
            sample["mask_path"]
        ).convert("L")

        # Transform image
        if self.image_transform:

            image = self.image_transform(image)

        # Transform mask
        if self.mask_transform:

            mask = self.mask_transform(mask)

        # Ensure binary mask
        mask = (mask > 0.5).float()

        return image, mask