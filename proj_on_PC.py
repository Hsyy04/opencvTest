from concurrent.futures import thread
from distutils.log import debug
import enum
from threading import Thread
from turtle import color

from transformers import TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
import cv2
import random
import time
from matplotlib import pyplot as plt

from paddle import save
import numpy as np
import math
from tensorboard import summary
import sys
from segment import segment, Frame

point = [100.0]
time_pt = []
clear = []

class Clarity_Filter:
    '''
        选择清晰的片段，删除运动过快的片段
    '''
    def __init__(self, time: float) -> None:
        self.time = time
        self.threshold = 0
        self.decay = 0.99

    def __call__(self, seg:segment, time: float) -> bool:
        dt = abs(time - self.time)
        self.threshold *= math.pow(self.decay, dt)
        self.time = time

        c = seg.clear
        if c >= self.threshold:
            self.threshold = c
            return True
        else:
            return False

class analyzer_thread(Thread):
    '''
        用于模拟手机上的实时处理。
    '''
    def __init__(self, func, args=()):
        super(analyzer_thread, self).__init__() # 其实没见过这种写法
        self.func = func
        self.args = args

    def run(self):
        self.func(self.args)

def initVideo(fps, path):
    cap = cv2.VideoCapture(path)
    # cap.set(5, fps)
    print(cap.get(cv2.CAP_PROP_FPS))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def getRand(fps, time, cap):
    summary_rand = []
    seg_ment=[]
    fr_cnt =0
    p = fps*time / cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        ret, frame = cap.read()
        if ret == False: 
            break
        fr_cnt+=1
        seg_ment.append(frame)
        if fr_cnt % fps ==0:
            rand = random.random()
            if rand <= p:
                summary_rand.extend(seg_ment)
            seg_ment.clear()
    return summary_rand

def saveVideo(summary, fps, size, type):
    print("save one!")
    t = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    out = cv2.VideoWriter(f"video/{type}{t}.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), fps, size, True)
    for fr in summary:
        out.write(cv2.resize(fr, size))
    out.release()

class hog_diff_dist:
    def __init__(self, cap, fps, time, seg_cnt) -> None:
        self.summary:list[segment] = []
        self.cap = cap
        self.fps = fps
        self.time = time
        self.bestdist = None
        self.threshold = 0.0
        self.segmetLength = time/seg_cnt
        self.seg_cnt = seg_cnt

    def HOG_diff_dist(self, now_segment:segment, usedDacy=False):
        if len(self.summary) < self.seg_cnt*3:
        # 为了防止摘要什么都没有，开始的片段直接加进来
            self.summary.append(now_segment)
        else:
            if self.bestdist == None:
                self.bestdist = 0.0
                for i in range(len(self.summary)-1):
                    self.bestdist += self.summary[i].hog_dist(self.summary[i+1])
                self.bestdist /= (len(self.summary)-1)
                self.threshold = self.bestdist
            # 是否重复
            last = self.summary[len(self.summary)-1]
            testdist = now_segment.hog_dist(last)
            

            # print(f"test_dist:{testdist}; bset: {self.bestdist}")
            if testdist < self.threshold:
                if usedDacy ==True : self.threshold*=0.9
                return 
            # 找替换后最好的
            k_frame = -1
            tem_dist = 100000000.0
            for k in range(len(self.summary)):
                t=0.0
                if k == 0:
                    t = self.summary[k].hog_dist(self.summary[k+1])
                elif k == len(self.summary)-1:
                    t = testdist + self.summary[k].hog_dist(self.summary[k-1])
                    t -= self.summary[k-1].hog_dist(now_segment)
                else:
                    t = self.summary[k].hog_dist(self.summary[k-1])+self.summary[k].hog_dist(self.summary[k+1])
                    t-= self.summary[k-1].hog_dist(self.summary[k+1])
                if tem_dist > t:
                    k_frame = k
                    tem_dist = t   
            
            # print(f"tem_dist:{tem_dist}, testdist:{testdist}")
            tem_dist= testdist-tem_dist
            if tem_dist < 10000000.0:
                self.bestdist+= tem_dist/(len(self.summary)-2)
                self.threshold = self.bestdist
            # 替换
                self.summary.pop(k_frame)
                self.summary.append(now_segment)

    def getSimple(self, debug=False, usedLK = False, usedDecay = False):
        frame_cnt = 0
        tem_segment = []
        ana_cnt = 0
        analyze_thread = None
        front = None
        seg = None

        # 使用光流分段
        if usedLK:
            ret, old_frame = cap.read()
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        while True:
            ret, frame = self.cap.read()  # 获取当前帧
            if ret == False:   # 视频结束
                break
            frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

            if analyze_thread!=None and analyze_thread.is_alive():  # 如果还在计算，那么得不到这一帧
                print(f"drop frame!{ana_cnt}---{cap.get(cv2.CAP_PROP_POS_FRAMES)}")
                continue
            
            # 帧计数，以及换存
            frame_cnt+=1
            if debug:   
                tem_segment.append(Frame(frame,idx=frame_cnt))
            else: 
                tem_segment.append(Frame(frame))


            # 计算新帧的光流， 如果光流需要重置， 就分段
            isClip = False
            if usedLK:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                isClip = (st is None) or (len(st)<10)
            else: 
                isClip = (frame_cnt % (self.fps*self.segmetLength)==0)
            if isClip :
                # print(f"this segment has{len(tem_segment)} frames")
                ana_cnt +=1
                seg = segment(tem_segment.copy())

                # 重建光流
                if usedLK:
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
                    old_gray = frame_gray.copy()

                if Clarity_Filter(seg) == False:
                    print(f"These are so moving!")
                    tem_segment.clear()
                    continue
                analyze_thread = analyzer_thread(self.HOG_diff_dist, seg)
                analyze_thread.start()
                tem_segment.clear()
            else:
                # 光流迭代
                if usedLK:
                    good_new = p1[st==1]
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1,1,2)
            
            if frame_cnt % 1000 == 0 :
                print(f"{frame_cnt}/{self.cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

        return self.summary

    def aestheticsChosen(self):
        # config
        nSegment = len(self.summary)
        nFrame = self.time * self.fps
        
        ret = []
        all_frame:list[Frame] = []
        for idx, fr in enumerate(self.summary):
            for i in fr.totDate:
                i.pos = len(all_frame)
                all_frame.append(i)
                i.belongWhichSegment = idx
        all_frame.sort(key = lambda x:x.color_score+x.symmetry_score, reverse=True)
        # 
        cntChosen = 0
        cntBeauty = [0 for i in range(nSegment)]
        retId = []
        for f in all_frame:
            # print(f"score:{f.color_score};{f.symmetry_score}\t seg:{f.belongWhichSegment}")
            if(cntChosen>=nFrame ):
                break
                # continue
            else:
                nowSeg = f.belongWhichSegment
                if cntBeauty[nowSeg] == -1: continue
                cntBeauty[nowSeg]+=1
                if cntBeauty[nowSeg] >= len(self.summary[nowSeg])/4.0:
                    retId.append(nowSeg)
                    cntChosen += len(self.summary[nowSeg])
                    cntBeauty[nowSeg] = -1

        retId.sort()
        ret = [self.summary[i] for i in retId]
        return ret

FPS = 16.0
TIME = 2       # 镜头时间
SIZE = (480,720)
SegCnt = 5      # 镜头个数
PATH = "video/test3.mp4"
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 1,
                       blockSize = 7 )
if __name__ == "__main__":

    cap = cv2.VideoCapture(PATH)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    print(f"time:{cap.get(cv2.CAP_PROP_FRAME_COUNT)/FPS}")

    print("calculating")
    ana = hog_diff_dist(cap, FPS, TIME*SegCnt, SegCnt)
    summary = ana.getSimple()
    save = []
    for s in summary:
        save.extend(s.frames)
    saveVideo(save, FPS, SIZE, "first_stage")

    summary = ana.aestheticsChosen()
    print("saving")
    save = []
    for s in summary:
        save.extend(s.frames)
    saveVideo(save, FPS, SIZE, "second_stage")

    # # save rand
    # rand = getRand(FPS, TIME, cap)
    # saveVideo(rand, FPS, SIZE, "rand")

    print("closing...")
    cap.release()
