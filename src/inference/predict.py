"""
RoadVision AI - Model Inference

Loads the trained RoadVision U-Net model and converts
satellite imagery into a binary road segmentation mask.
"""

import os

import torch
from torchvision import transforms
from PIL import Image

from src.models.unet import UNet


# ============================================================
# FINAL MODEL
# ============================================================

MODEL_PATH = "src/models/ROADVISION_FINAL_BEST.pth"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path=MODEL_PATH):
    """
    Load the RoadVision U-Net model.

    Supports both:
        1. Raw state_dict checkpoints
        2. Checkpoints containing a 'state_dict' key
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = UNet()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# PREDICTION
# ============================================================

def predict(
    image_path,
    model_path=MODEL_PATH
):
    """
    Generate a binary road segmentation mask.

    Parameters
    ----------
    image_path : str
        Input satellite image.

    model_path : str
        Path to trained RoadVision model.

    Returns
    -------
    mask : torch.Tensor
        Binary road mask.

    original_size : tuple
        Original image dimensions.
    """

    model, device = load_model(
        model_path
    )

    # --------------------------------------------------------
    # Same preprocessing used during final model inference
    # --------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize(
            (256, 256),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    original_size = image.size

    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        )

    # --------------------------------------------------------
    # Convert probability to binary mask
    # --------------------------------------------------------

    mask = (
        probability > 0.5
    ).float()

    mask = mask.squeeze().cpu()

    return mask, original_size