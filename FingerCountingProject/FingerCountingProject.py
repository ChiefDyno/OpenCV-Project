import cv2 as cv
import time
import os
import numpy as np
import HandTrackingModule as htm

##########################
wCam, hCam = 1000, 1200
###########################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "fingers"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))

pTime = 0
cTime = 0

detector = htm.handDetector()
tipId = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    img = cv.flip(img, 1)
    lmList, bbox = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        finger = []
        # thumb
        if lmList[tipId[0]][1] > lmList[tipId[0] - 1][1]:
            finger.append(1)
        else:
            finger.append(0)

        for ID in range(1, 5):
            if lmList[tipId[ID]][2] < lmList[tipId[ID] - 2][2]:
                finger.append(1)
            else:
                finger.append(0)

        totalFinger = finger.count(1)
        print(totalFinger)

        h, w, c = overlayList[0].shape
        img[0:h, 0:w] = overlayList[totalFinger - 1]

        cv.rectangle(img, (40, 370), (160, 480), (0, 255, 0), thickness=cv.FILLED)
        cv.putText(img, str(totalFinger), (80, 450), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)

        # print(finger)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{str(int(fps))}', (700, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=3)

    cv.imshow("Image", img)

    if cv.waitKey(1) == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
