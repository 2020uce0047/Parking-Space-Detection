import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

class transformClass():
    def __init__(self, img):
        self.img = img
        self.data_transform = transforms.Compose([
         transforms.Resize(size = (64, 64)),

         transforms.RandomHorizontalFlip(p = 0.5),

         transforms.ToTensor()])
    def transform(self):
         return self.data_transform(self.img)
    

# img = cv2.imread('Image/sample.png')
# img = Image.fromarray(img)
# t = transformClass(img).transform()
# cv2.imshow('Transformed Image', np.array(t.permute(1,2,0)))
# cv2.waitKey(0)
