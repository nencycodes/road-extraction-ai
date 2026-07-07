import cv2


def detect_edges(image):

    edges = cv2.Canny(
        image,
        threshold1=100,
        threshold2=200
    )

    return edges