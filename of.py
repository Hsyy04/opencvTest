import numpy as np
import cv2
import time

PATH = "video/hog_diff_clear2022_04_12_19_44_17.mp4"

cap = cv2.VideoCapture(PATH)
t = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
out = cv2.VideoWriter(f"video/optical_flow{t}.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 16, (480,720), True)

# ShiTomasi 角点检测参数
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 1,
                       blockSize = 7 )

# lucas kanade光流法参数
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机颜色
color = np.random.randint(0,255,(100,3))

# 获取第一帧，找到角点
ret, old_frame = cap.read()
#找到原始灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#获取图像中的角点，返回到p0中
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# 创建一个蒙版用来画轨迹
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    if ret == False:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if (st is None) or (len(st)<10):
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
        mask = np.zeros_like(old_frame)
        continue
    # 选取好的跟踪点
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    max_dist = 0.0
    # 画出轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        dis = (a-c)*(a-c)+(b-d)*(b-d)
        max_dist = max(dis, max_dist)
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.putText(img, str(max_dist), (140, 150), cv2.FONT_HERSHEY_PLAIN, 6.0, (0, 0, 255), 6)
    if max_dist<1500.0:
        out.write(cv2.resize(img, (480,720)))

    # 更新上一帧的图像和追踪点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cap.release()
out.release()