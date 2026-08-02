"""
road_dataset.py

Custom PyTorch Dataset for the Massachusetts Roads Dataset.
"""

import os

from PIL import Image

from torch.utils.data import Dataset


class RoadDataset(Dataset):
    """
    Custom Dataset for Road Segmentation.

    Returns:
        image : Tensor
        mask  : Tensor
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get all image names (without extension)
        image_names = {
            os.path.splitext(file)[0]
            for file in os.listdir(image_dir)
        }

        # Get all mask names (without extension)
        mask_names = {
            os.path.splitext(file)[0]
            for file in os.listdir(mask_dir)
        }

        # Keep only matching image-mask pairs
        self.valid_images = sorted(
            image_names & mask_names
        )

        print(
            f"Loaded {len(self.valid_images)} image-mask pairs."
        )

    def __len__(self):

        return len(self.valid_images)

    def __getitem__(self, index):

        image_name = self.valid_images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name + ".tiff"
        )

        mask_path = os.path.join(
            self.mask_dir,
            image_name + ".tif"
        )

        image = Image.open(image_path).convert("RGB")

        mask = Image.open(mask_path).convert("L")

        if self.transform:

            image = self.transform(image)

            mask = self.transform(mask)
            mask = (mask > 0).float()

        return image, mask