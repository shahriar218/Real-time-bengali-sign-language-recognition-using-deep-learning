import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
folder = "Data/Kha"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        # Create a bounding box that encloses both hands
        x, y, w, h = hands[0]['bbox']
        for hand in hands:
            if hand['bbox'][0] < x:
                w += x - hand['bbox'][0]
                x = hand['bbox'][0]
            if hand['bbox'][1] < y:
                h += y - hand['bbox'][1]
                y = hand['bbox'][1]
            if hand['bbox'][0] + hand['bbox'][2] > x + w:
                w = hand['bbox'][0] + hand['bbox'][2] - x
            if hand['bbox'][1] + hand['bbox'][3] > y + h:
                h = hand['bbox'][1] + hand['bbox'][3] - y

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
