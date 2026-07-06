import cv2


def read_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read image.")

    return image