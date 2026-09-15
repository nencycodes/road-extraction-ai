import cv2


def preprocess_image(image):
    """
    Basic image preprocessing for the classical
    image-processing pipeline.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    resized = cv2.resize(
        image,
        (512, 512)
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return blurred