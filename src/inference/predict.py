import torch
from torchvision import transforms
from PIL import Image

from src.models.unet import UNet


MODEL_PATH = "models/best_model_v1_256.pth"


def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNet()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model, device


def predict(image_path):
    model, device = load_model()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output)

    mask = (probability > 0.5).float()

    mask = mask.squeeze().cpu()

    return mask, original_size