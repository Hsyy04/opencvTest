import cv2
from cv2 import imread
from cv2 import waitKey

img = imread('img/maomao.jpg')
img = cv2.resize(img, (480,720))
img_flip = cv2.flip(img,1)
cv2.imshow('origin',img)
waitKey(5000)
cv2.imshow('symmetry',img_flip)
waitKey(5000)

sift = cv2.SIFT_create(10)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift0=sift.detectAndCompute(img,None)
print(sift0)

img_flip = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)
sift1=sift.detectAndCompute(img_flip,None)
print(sift1)

