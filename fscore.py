from segment import Frame, segment
from summe import evaluateSummary, plotAllResults
import os
import random
import scipy.io
import numpy as np
import cv2
from proj_on_PC import hog_diff_dist


def randomSummary(videoName, partion):                    
    
    #In this example we need to do this to now how long the summary selection needs to be
    gt_file=HOMEDATA+'/'+videoName+'.mat'
    gt_data = scipy.io.loadmat(gt_file)
    nFrames=gt_data.get('nFrames')
    '''Example summary vector''' 
    #selected frames set to n (where n is the rank of selection) and the rest to 0
    summary_selections=np.random.random((int(nFrames),1))*20
    summary_selections=list(map(lambda q: (round(q[0]) if (q >= np.percentile(list(summary_selections),100-partion*100)) else 0),summary_selections))
    return summary_selections

def fovlogSummary(videoName):

    gt_file=HOMEDATA+'/'+videoName+'.mat'
    gt_data = scipy.io.loadmat(gt_file)
    nFrames=gt_data.get('nFrames')

    FPS = 16.0
    TIME = 2       # 镜头时间
    SIZE = (480,720)
    SegCnt = 15

    video_file=HOMEVIDEOS+'/'+videoName+'.mp4'
    cap = cv2.VideoCapture(video_file)
    ana = hog_diff_dist(cap, FPS, TIME*SegCnt, SegCnt)
    summary:list[segment] = ana.getSimple(debug=True)
    summary = ana.aestheticsChosen()

    FrameSet = []
    for seg in summary:
        for fr in seg.totDate:
            FrameSet.append(fr.raw_id)

    ret_summary = []
    for i in range(nFrames[0][0]):
        if i in FrameSet: ret_summary.append(1)
        else: ret_summary.append(0)
    
    print("closing...")
    cap.release()
    return ret_summary

if __name__ == "__main__":
    HOMEDATA='summeData/GT/'
    HOMEVIDEOS='summeData/videos/'

    included_extenstions=['mp4']
    videoList=[fn for fn in os.listdir(HOMEVIDEOS) if any([fn.endswith(ext) for ext in included_extenstions])]
    videoNameList = [videoName.split('.')[0] for videoName in videoList]

    for name in videoNameList:
        print(f"-----{name}-----")
        smp = fovlogSummary(name)
        print("simple")
        [f_measure,summary_length]=evaluateSummary(smp, name, HOMEDATA)
        print('F-measure : %.3f at length %.2f' % (f_measure, summary_length))
        rd = randomSummary(name, summary_length)
        print("random:")
        [f_measure,summary_length]=evaluateSummary(rd,name,HOMEDATA)
        print('F-measure : %.3f at length %.2f' % (f_measure, summary_length))
        print("\n")
