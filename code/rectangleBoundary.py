import numpy as np
import cv2
from PIL import Image
import math
import pandas as pd

height, width = 25, 65
# x, y = 300, 300
img = cv2.imread('Image/sample.png', 1)
positions = []
from os.path import exists


def click_event(events, x, y, flags, params):
    if(events == cv2.EVENT_LBUTTONDOWN):
        positions.append((x, y))
        cv2.rectangle(img, (x, y), (x+width, y+height), color = (0, 255, 0), thickness=2)
        # file_exists = exists('posList.csv')
        # if(file_exists):
        #     print('File already exists!')

        df = pd.DataFrame({'Co-ordinates': positions})
        df.to_csv('posList.csv')
while True:
    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
    # cv2.rectangle(img, (x, y), (x+width, y+height), color = (0, 0, 255))
    
    # wait for a key to be pressed to exit
    cv2.waitKey(1)



