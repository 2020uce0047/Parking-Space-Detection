import cv2
import cvzone
import numpy as np
import pandas as pd
 
cap = cv2.VideoCapture('Videos/parking_1920_1080_loop.mp4')
img = cv2.imread('Image/sample.png', 1)
posList = pd.read_csv('posList.csv')['Co-ordinates']
width, height = 65, 25
 

def empty(a):
    pass


cv2.namedWindow("Vals")
cv2.resizeWindow("Vals", 640, 240)
cv2.createTrackbar("Val1", "Vals", 25, 50, empty)
cv2.createTrackbar("Val2", "Vals", 16, 50, empty)
cv2.createTrackbar("Val3", "Vals", 5, 50, empty)
 

def checkSpaces():
    spaces = 0
    for pos in posList:
        x, y = pos[1:len(pos)-1].split(', ')
        x = int(x)
        y = int(y)
 
        imgCrop = imgThres[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
 
        if count < 500:
            color = (0, 255, 0)
            thic = 5
            spaces += 1
 
        else:
            color = (0, 0, 255)
            thic = 2
 
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thic)
 
        # cv2.putText(img, str(cv2.countNonZero(imgCrop)), (x, y + h - 6), cv2.FONT_HERSHEY_PLAIN, 1,
        #             color, 2)
 
    # cvzone.putTextRect(img, f'Free: {spaces}/{len(posList)}', (50, 60), thickness=3, offset=20,
    #                    colorR=(0, 200, 0))
 
 
while True:
 
    # Get image frame
    success, img = cap.read()
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # img = cv2.imread('img.png')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    # ret, imgThres = cv2.threshold(imgBlur, 150, 255, cv2.THRESH_BINARY)
 
    val1 = cv2.getTrackbarPos("Val1", "Vals")
    val2 = cv2.getTrackbarPos("Val2", "Vals")
    val3 = cv2.getTrackbarPos("Val3", "Vals")
    if val1 % 2 == 0: val1 += 1
    if val3 % 2 == 0: val3 += 1
    imgThres = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, val1, val2)
    imgThres = cv2.medianBlur(imgThres, val3)
    kernel = np.ones((3, 3), np.uint8)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)
 
    checkSpaces()
    # Display Output
 
    cv2.imshow("Image", img)
    # cv2.imshow("ImageGray", imgThres)
    # cv2.imshow("ImageBlur", imgBlur)
    key = cv2.waitKey(1)
    if key == ord('r'):
        pass
