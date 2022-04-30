from concurrent.futures import thread
import enum
from threading import Thread
from turtle import color
import cv2
import random
import time
from matplotlib import pyplot as plt

from paddle import save
import numpy as np

from tensorboard import summary
import sys
from segment import segment

point = [100.0]
time_pt = []
clear = []

class analyzer_thread(Thread):
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

class hog_whole_dist:
    def __init__(self, cap, fps, time) -> None:
        self.summary = []
        self.cap = cap
        self.fps = fps
        self.time = time
        self.bestdist = None

    def HOG_whole_dist(self, now_segment:segment):
        if len(self.summary) < self.time:
        # 为了防止摘要什么都没有，开始的片段直接加进来
            self.summary.append(now_segment)
        else:
            if self.bestdist == None:
                self.bestdist = 0.0
                for i, seg in enumerate(self.summary):
                    for j in range(i+1, len(self.summary)):
                        self.bestdist+= seg.hog_dist(self.summary[j])
                        # print(f"init_dist:{seg.hog_dist(self.summary[j])}")
                self.bestdist /= self.time*(self.time-1)
            # 是否重复
            last = self.summary[len(self.summary)-1]
            testdist = now_segment.hog_dist(last)
            # print(f"test_dist:{testdist}")

            if testdist/2.0 < 2000.0:
                if now_segment.clearFeature > last.clearFeature:
                    self.summary.pop()
                    self.summary.append(now_segment)
                    # update best
                    self.bestdist = 0.0
                    for i, seg in enumerate(self.summary):
                        for j in range(i+1, len(self.summary)):
                            self.bestdist+= seg.hog_dist(self.summary[j])
                            # print(f"init_dist:{seg.hog_dist(self.summary[j])}")
                    self.bestdist /= self.time*(self.time-1)
                return 
            # 找替换后最好的
            k_frame = -1
            for k in range(len(self.summary)):
                tem_dist = 0.0
                for i in range(len(self.summary)):
                    for j in range(i+1, len(self.summary)):
                        if i == k:
                            tem_dist+=self.summary[j].hog_dist(now_segment)
                        elif j == k:
                            tem_dist+=self.summary[i].hog_dist(now_segment)
                        else:
                            tem_dist+=self.summary[i].hog_dist(self.summary[j])
                tem_dist /= self.time*(self.time-1)
                # print(f"segment:{k}, dist:{tem_dist}, best_dist:{self.bestdist}")
                if tem_dist > self.bestdist:
                    k_frame = k
                    self.bestdist = tem_dist
            # 替换
            if k_frame != -1:
                self.summary.pop(k_frame)
                self.summary.append(now_segment)

    def getSimple1(self):
        frame_cnt = 0
        tem_segment = []
        ana_cnt = 0
        analyze_thread = None
        front = None
        seg = None
        while True:
            ret, frame = self.cap.read()
            if ret == False: 
                break
            if analyze_thread!=None and analyze_thread.is_alive():
                print(f"drop frame!{ana_cnt}---{cv2.CAP_PROP_POS_FRAMES}")
                continue
            frame_cnt+=1
            tem_segment.append(frame)
            if frame_cnt % self.fps == 0:
                ana_cnt +=1
                front = seg 
                seg = segment(tem_segment.copy())
                if seg.clearFeature < 2500:
                    tem_segment.clear()
                    continue
                analyze_thread = analyzer_thread(self.HOG_whole_dist, seg)
                analyze_thread.start()
                if frame_cnt != self.fps:
                    point.append(seg.hog_dist(front))
                    time_pt.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)
                tem_segment.clear()
            
            if frame_cnt % 300 == 0:
                sys.stderr.write(f"{frame_cnt}/{cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

        return self.summary

class hog_diff_dist:
    def __init__(self, cap, fps, time) -> None:
        self.summary = []
        self.cap = cap
        self.fps = fps
        self.time = time
        self.bestdist = None

    def HOG_diff_dist(self, now_segment:segment):
        if len(self.summary) < self.time:
        # 为了防止摘要什么都没有，开始的片段直接加进来
            self.summary.append(now_segment)
        else:
            if self.bestdist == None:
                self.bestdist = 0.0
                for i in range(len(self.summary)-1):
                    self.bestdist += self.summary[i].hog_dist(self.summary[i+1])
                self.bestdist /= (self.time-1)
            # 是否重复
            last = self.summary[len(self.summary)-1]
            testdist = now_segment.hog_dist(last)
            # print(f"test_dist:{testdist}; bset: {self.bestdist}")

            if testdist < self.bestdist:
                if now_segment.clearFeature > last.clearFeature:
                    self.summary.pop()
                    self.summary.append(now_segment)
                    # update best
                    self.bestdist = 0.0
                    for i in range(len(self.summary)-1):
                        self.bestdist += self.summary[i].hog_dist(self.summary[i+1])
                    self.bestdist /= (self.time-1)
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
                self.bestdist+= tem_dist/(self.time-1)
            # 替换
                self.summary.pop(k_frame)
                self.summary.append(now_segment)

    def getSimple1(self):
        frame_cnt = 0
        tem_segment = []
        ana_cnt = 0
        analyze_thread = None
        front = None
        seg = None

        # 使用光流分段
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        while True:
            ret, frame = self.cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if ret == False: 
                break

            if analyze_thread!=None and analyze_thread.is_alive():
                print(f"drop frame!{ana_cnt}---{cv2.CAP_PROP_POS_FRAMES}")
                continue

            frame_cnt+=1
            tem_segment.append(frame)

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if (st is None) or (len(st)<10) :
                print(f"this segment has{len(tem_segment)} frames")
                ana_cnt +=1
                front = seg 
                seg = segment(tem_segment.copy())

                if seg.clearFeature < 2500:
                    tem_segment.clear()
                    continue
                analyze_thread = analyzer_thread(self.HOG_diff_dist, seg)
                analyze_thread.start()
                tem_segment.clear()
                # 重建光流
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
                old_gray = frame_gray.copy()
                st.clear()
            else:
                # 光流迭代
                good_new = p1[st==1]
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            
            if frame_cnt % 300 == 0:
                print(f"{frame_cnt}/{self.cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

        return self.summary

FPS = 16.0
TIME = 20
SIZE = (480,720)
PATH = "video/test1.mp4"
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 1,
                       blockSize = 7 )

cap = cv2.VideoCapture(PATH)
cap.set(cv2.CAP_PROP_FPS, FPS)
print(f"time:{cap.get(cv2.CAP_PROP_FRAME_COUNT)/15.0}")

print("calculating")
ana = hog_diff_dist(cap, FPS, TIME)
summary = ana.getSimple1()


print("saving")
save = []
for s in summary:
    save.extend(s.frames)

saveVideo(save, FPS, SIZE, "hog_whole_clear")

# # save rand
# rand = getRand(FPS, TIME, cap)
# saveVideo(rand, FPS, SIZE, "rand")

print("closing...")
cap.release()
