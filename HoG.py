from cv2 import COLOR_RGB2GRAY, HOGDescriptor, imshow
import cv2 


def getFeature(img:cv2.Mat, winSize=(128,128), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9):
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins) 
    hist = hog.compute(img[0:128,0:128])
    print(type(hist))
    return hist


img = cv2.imread("img/tutu.jpg")
print(getFeature(img).size)
# imshow("img",img)
# cv.waitKey(0)