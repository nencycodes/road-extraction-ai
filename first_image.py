import cv2
import matplotlib.pyplot as plt
img = cv2.imread("images/Road_image1.jpg")
#inside image is A NumPy Array. 
# here shape = dimensions of the image in array format.
print("Shape:", img.shape)
# it gave (408,612,3)
# means 408 rows = height of the image
#612 columns = width
# 3 color values = RGB Colours(open cv2 used BGR Format)
#this image has 408* 612 pixels = 249,696 pixels
print("First Pixel:", img[0][0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Gray Shape:", gray.shape)
plt.imshow(gray, cmap='gray')
plt.show()