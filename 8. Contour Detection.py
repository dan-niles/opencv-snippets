import cv2
import numpy as np
from helpers.stackImages import stackImages


def getContours(img):
    pass


img = cv2.imread("img/shapes.png")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)

imgCanny = cv2.Canny(imgBlur, 50, 50)

# cv2.imshow("Image", img)
# cv2.imshow("Image Gray", imgGray)
# cv2.imshow("Image Blur", imgBlur)

imgBlank = np.zeros_like(img)

imgStack = stackImages(0.6, ([img, imgGray, imgBlur], [imgCanny, imgBlank, imgBlank]))
cv2.imshow("Image Stack", imgStack)
cv2.waitKey(0)
