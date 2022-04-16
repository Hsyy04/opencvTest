import cv2
import numpy as np

img = cv2.imread('img/tutu.jpg')

result3 = img.copy()

img = cv2.GaussianBlur(img,(3,3),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imwrite("img/canny.jpg", edges)

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
m = cv2.getPerspectiveTransform(src, dst)
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("result", result)

cv2.waitKey(0)