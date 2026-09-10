import os

import torch
from torchvision import transforms
from PIL import Image

from src.models.unet import UNet


# Default model.
# Change this to the final aggressive model when it is copied
# into src/models/.
MODEL_PATH = "src/models/best_model_v1_256.pth"


def load_model(model_path=MODEL_PATH):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
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

    # Supports a raw state_dict or a wrapped checkpoint.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.to(device)
    model.eval()

    return model, device


def predict(image_path, model_path=MODEL_PATH):

    model, device = load_model(model_path)

    transform = transforms.Compose([
        transforms.Resize(
            (256, 256),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    original_size = image.size

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        )

    mask = (
        probability > 0.5
    ).float()

    mask = mask.squeeze().cpu()

    return mask, original_size