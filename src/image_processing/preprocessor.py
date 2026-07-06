import cv2


def preprocess_image(image):

    resized = cv2.resize(image, (512, 512))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred