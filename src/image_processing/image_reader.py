import cv2


def read_image(image_path):
    """
    Read an image from disk using OpenCV.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image