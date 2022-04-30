from typing import List
import cv2
import numpy as np
from c3d_feature import C3D_feature
import torch
import torchvision

class segment:
    def __init__(self, frames) -> None:
        # assert len(frames)>=16
        self.frames = frames
        self.hogFeature = []
        self.clearFeature = []
        for fr in frames:
            self.hogFeature.append(getHogFeature(fr))
            self.clearFeature.append(getClearFeature(fr))
        self.hogFeature = np.array(self.hogFeature)
        self.hogFeature = np.average(self.hogFeature, axis=1)
        self.clearFeature = np.array(self.clearFeature).mean()

        # model = C3D_feature().eval()
        # input_size = (224,112)
        # input = [cv2.resize(i, input_size) for i in self.frames]
        # input = torch.Tensor(input)
        # input = torch.transpose(input, -1, 1).unsqueeze(0).transpose(1,2)
        
        # self.c3dFeature = model(input).reshape(-1)

        # model = torchvision.models.mobilenet_v2(pretrained=True).eval()
        # input_size = (224,112)
        # input = [cv2.resize(i, input_size) for i in self.frames]
        # input = torch.Tensor(input)
        # input = torch.transpose(input, -1, 1)
        # self.mbFeature = model(input)
        # print(self.mbFeature.shape)

    def hog_dist(self, obj):
        return cv2.norm(self.hogFeature - obj.hogFeature, 2)
    
def getClearFeature(src):
    cal_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(cal_img, 200, 200)
    return canny.var()

def getHogFeature(img:cv2.Mat, winSize=(64,128), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9):
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    cal_img = cv2.cvtColor(cv2.resize(img, winSize), cv2.COLOR_BGR2GRAY)
    hist = hog.compute(cal_img)
    return hist

def getSIFTFeature(img:cv2.Mat):

    pass

