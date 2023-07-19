import cv2

frameWidth = 640
frameHeight = 480
nPlateCascadePath = "/Users/danniles/miniforge3/pkgs/libopencv-4.7.0-py39h4b58551_1/share/opencv4/haarcascades/haarcascade_russian_plate_number.xml"
nPlateCascade = cv2.CascadeClassifier(nPlateCascadePath)
minArea = 500
color = (255, 0, 255)

cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 100)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    for x, y, w, h in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w + 10, y + h + 10), color, 2)
            cv2.putText(
                img,
                "Number Plate",
                (x, y - 5),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                color,
                2,
            )
            imgRoi = img[y : y + h, x : x + w]

    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
