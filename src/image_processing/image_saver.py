import cv2
from pathlib import Path

PROCESSED_FOLDER = Path("data/processed")
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


def save_processed_image(image, filename):

    save_path = PROCESSED_FOLDER / filename

    cv2.imwrite(str(save_path), image)

    return save_path