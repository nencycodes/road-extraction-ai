import cv2


def detect_edges(image):
# canny function helps to detect the changes in the image and detect the edges of the image
    edges = cv2.Canny(
        image,
        threshold1=100,
        threshold2=200
    )

    return edges