import cv2
import numpy as np
from helpers.stackImages import stackImages

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 100)


def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDialation, kernel, iterations=1)

    return imgThres


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (0, 255, 0), 20)
    return biggest


def reorder(myPoints):
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    myPoints = myPoints.reshape((4, 2))

    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def getWarp(img, biggest):
    newPoints = reorder(biggest)
    width, height = frameWidth, frameHeight

    pts1 = np.float32(newPoints)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    imgCropped = imgOutput[20 : imgOutput.shape[0] - 20, 20 : imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (width, height))

    return imgCropped


while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    imgContour = img.copy()

    imgThres = preprocessing(img)
    biggest = getContours(imgThres)

    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        imgArray = ([img, imgThres], [imgContour, imgWarped])
    else:
        imgArray = ([img, imgThres], [img, img])

    imgStacked = stackImages(0.6, imgArray)

    cv2.imshow("Stacked Output", imgStacked)
    cv2.imshow("Result", imgWarped)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
