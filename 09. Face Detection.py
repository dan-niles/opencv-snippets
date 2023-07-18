import cv2
import numpy as np

faceCascadePath = "/Users/danniles/miniforge3/pkgs/libopencv-4.7.0-py39h4b58551_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(faceCascadePath)
img = cv2.imread("img/face.jpeg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w + 10, y + h + 10), (255, 0, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
