import cv2
import numpy as np

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


def getWarp(img, biggest):
    width, height = frameWidth, frameHeight

    pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    pass


while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    imgContour = img.copy()

    imgThres = preprocessing(img)
    biggest = getContours(imgThres)
    getWarp(img, biggest)

    cv2.imshow("video", imgContour)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
