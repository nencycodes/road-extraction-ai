import cv2


def detect_edges(image):
    """
    Detect image edges using the Canny edge detector.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    edges = cv2.Canny(
        image,
        threshold1=100,
        threshold2=200
    )

    return edges