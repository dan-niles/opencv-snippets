import cv2
import numpy as np

img = cv2.imread("img/sports_car.png")
print(img.shape)

imgResize = cv2.resize(img, (300, 200))
imgCropped = img[0:200, 100:150]  # Height comes first

cv2.imshow("Image", img)
cv2.imshow("Image Resized", imgResize)
cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)
