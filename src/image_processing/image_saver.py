import cv2
from pathlib import Path


PROCESSED_FOLDER = Path("data/processed")
EDGE_FOLDER = Path("data/edges")

PROCESSED_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

EDGE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def save_processed_image(image, filename):
    """
    Save a processed image.
    """

    save_path = PROCESSED_FOLDER / filename

    success = cv2.imwrite(
        str(save_path),
        image
    )

    if not success:
        raise IOError(
            f"Unable to save processed image: {save_path}"
        )

    return save_path


def save_edge_image(image, filename):
    """
    Save an edge-detection image.
    """

    save_path = EDGE_FOLDER / filename

    success = cv2.imwrite(
        str(save_path),
        image
    )

    if not success:
        raise IOError(
            f"Unable to save edge image: {save_path}"
        )

    return save_path