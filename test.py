import cv2
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2) # set maxHands to 2
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 300
folder = "Data/OI"
counter = 0
labels = ["A", "I", "Ka", "Kha", "La", "Ma", "O", "OI", "Pa", "Ra"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        x_min, y_min = imgSize, imgSize
        x_max, y_max = 0, 0
        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x - offset)
            y_min = min(y_min, y - offset)
            x_max = max(x_max, x + w + offset)
            y_max = max(y_max, y + h + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y_min:y_max, x_min:x_max]
        imgCropShape = imgCrop.shape

        aspectRatio = imgCropShape[0] / imgCropShape[1]
        if aspectRatio > 1:
            k = imgSize / imgCropShape[0]
            wCal = math.ceil(k * imgCropShape[1])
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        else:
            k = imgSize / imgCropShape[1]
            hCal = math.ceil(k * imgCropShape[0])
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x_min, y_min - offset - 50), (x_max, y_min - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x_min, y_min - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
