import numpy as np
import cv2
from PIL import Image
import math
import pandas as pd
import torch
from transform_img import transformClass
from model import PKLot_modelV1
from torchvision import transforms


cap = cv2.VideoCapture('Videos/parking_1920_1080_loop.mp4')
MODEL_SAVE_PATH = 'model/PKLot.pth'
trainedModel = PKLot_modelV1(input_channels = 3, hidden_units = 10, output_channels = 2)
trainedModel.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=torch.device('cpu')))
# print(trainedModel)
height, width = 25, 65
# x, y = 300, 300
img = cv2.imread('Image/sample.png', 1)
posList = pd.read_csv('posList.csv')['Co-ordinates']
class_names = ['Empty', 'Occupied']



data_transform = transforms.Compose([
         transforms.Resize(size = (64, 64)),

         transforms.RandomHorizontalFlip(p = 0.5),

         transforms.ToTensor()])
while True:
    # displaying the image
        # Get image frame
    success, img = cap.read()
   

    

    for pos in posList:
        x, y = pos[1:len(pos)-1].split(', ')
        x = int(x)
        y = int(y)
        # crop_img = img[y:y+height, x:x+width]
        # print(type(crop_img))
        crop_img = Image.fromarray(img[y:y + height, x:x + width])
        # crop_img = crop_img.crop((x, y, x + width, y + height))
        
        transformed_image = data_transform(crop_img)
        trainedModel.eval()
        with torch.inference_mode():
            y_pred = trainedModel(transformed_image.unsqueeze(dim = 0))
        parkingStatus = class_names[y_pred.argmax(dim = 1)[0]]
        if(parkingStatus == 'Empty'):
            cv2.rectangle(img, (x, y), (x+width, y+height), color = (0, 255, 0), thickness=2)
        else:
            cv2.rectangle(img, (x, y), (x+width, y+height), color = (0, 0, 255), thickness=2)




    # Display Output
    cv2.imshow("Image", img)
    
    # cv2.imshow("ImageGray", imgThres)
    # cv2.imshow("ImageBlur", imgBlur)
    key = cv2.waitKey(1)
    if key == ord('r'):
        pass
    
