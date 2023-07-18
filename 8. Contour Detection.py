import cv2
import numpy as np
from helpers.stackImages import stackImages


def getContours(img, imgContour):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (0, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3:
                objectType = "Tri"
            elif objCor == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4:
                objectType = "Circle"
            else:
                objectType = "None"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(
                imgContour,
                objectType,
                (x + (w // 2) - 10, y + (h // 2) - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )


img = cv2.imread("img/shapes.png")
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)

imgCanny = cv2.Canny(imgBlur, 50, 50)

getContours(imgCanny, imgContour)

imgBlank = np.zeros_like(img)

imgStack = stackImages(0.6, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgBlank]))
cv2.imshow("Image Stack", imgStack)
cv2.waitKey(0)
