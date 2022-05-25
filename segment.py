from typing import List
import cv2
import numpy as np
from c3d_feature import C3D_feature
import torch
import torchvision
import math

def get_dist(x, y, i, j):
    return math.sqrt(math.pow(x-i,2)+math.pow(y-j,2))
def get_simi(f1, f2):
    return torch.cosine_similarity(torch.from_numpy(f1),torch.from_numpy(f2), dim=0)

class Frame:
    def __init__(self, frame:cv2.Mat, idx=-1) -> None:
        self.frame = frame
        self.clear_score = self.getClearFeature(frame)
        # print(f"clear: {self.clear_score}")
        self.symmetry_score = self.getSymmetry(frame, 1)+self.getSymmetry(frame, 0)
        # print(f"symmetry:{self.symmetry_score}")
        # self.symmetry1_score = self.getSymmetry(frame, 1)
        # self.symmetry0_score = self.getSymmetry(frame, 0) 
        self.hog = self.getHogFeature(frame)
        self.color_score = self.getColor(frame)
        # print(f"color: {self.color_score}")
        self.pos = 0            # For aesthetic：最终摘要中的第几帧
        self.belongWhichSegment = 0         # For aesthetic: 属于第几个片段
        self.raw_id = idx # 原始视频中的第几帧，用于实验评分

    def getSymmetry(self, img:cv2.Mat, code:int):
        img_cal = cv2.resize(img,(270,480))
        # 创建对称图
        img_flip_src = cv2.flip(img_cal,code)
        # 将特征点对称
        # 创建sift计算子
        feature_point_size = 100
        sift = cv2.SIFT_create(feature_point_size)

        img_gray = cv2.cvtColor(img_cal, cv2.COLOR_BGR2GRAY)
        img_flip_gray = cv2.cvtColor(img_flip_src, cv2.COLOR_BGR2GRAY)
        # 获取该图像的特征点
        sift_src=sift.detectAndCompute(img_gray,None)
        sift_flip=sift.detectAndCompute(img_flip_gray,None)

        avg_sc = 0.0
        theta = 50.0
        for src_id, src_p in enumerate(sift_src[0]):
            x,y= src_p.pt[0], src_p.pt[1]
            dist = []
            simi = []
            for flip_id, flip_p in enumerate(sift_flip[0]):
                i,j = flip_p.pt[0], flip_p.pt[1]
                dis = get_dist(x,y,i,j)
                if dis <= theta:
                    dist.append(dis)
            if len(dist) != 0:
                dist = torch.log_softmax(torch.Tensor(dist),dim=0)
                sc =0-min(dist)
                avg_sc += sc/ float(feature_point_size)
        return avg_sc

    def getColor(self, img:cv2.Mat):
        img_src = cv2.resize(img, (25,50))
        img_hsv = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
        score = 0.0
        cnt =0
        for i,row in enumerate(img_hsv):
            for j,pixel in enumerate(row):
                cnt+=1
                sc = (float(pixel[1])*0.5+float(pixel[2]))*0.5/255.0
                score+= sc*100.0
        return score/(25.0*50.0)

    def getClearFeature(self, src):
        cal_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(cal_img, 200, 200)
        return canny.var()

    def getHogFeature(self, img:cv2.Mat, winSize=(64,128), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9):
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
        cal_img = cv2.cvtColor(cv2.resize(img, winSize), cv2.COLOR_BGR2GRAY)
        hist = hog.compute(cal_img)
        return hist

class segment:
    def __init__(self, frames:list[Frame]) -> None:
        # assert len(frames)>=16
        self.totDate:list[Frame] = frames
        self.frames = []
        self.clear = 0.0
        self.score1 = 0.0
        self.score2 = 0.0
        self.feature = []
        for fr in frames:
            self.frames.append(fr.frame)
            self.clear+=fr.clear_score
            self.feature.append(fr.hog)
            self.score1 += fr.symmetry_score
            self.score2 += fr.color_score
        self.clear/=float(len(self.frames)) # 视频片段的清晰度得分是每帧的平均值
        self.feature = np.average(np.array(self.feature), axis=0) # 视频片段的特征是每帧的平均值
        self.score1/=float(len(self.frames))
        self.score2/=float(len(self.frames))
        # print(self.feature.shape)


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

    def __len__(self):
        return len(self.frames)

    def hog_dist(self, obj):
        return cv2.norm(self.feature - obj.feature, 2)
    

