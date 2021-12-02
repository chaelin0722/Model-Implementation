import cv2, sys
from matplotlib import pyplot as plt
import numpy as np


image = cv2.imread('pic_thre_1_black.bmp')

blur = cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)


edged1 = cv2.Canny(image, 10,250)

edged = cv2.Canny(blur, 10, 250)
#cv2.imshow('Edged', edged)
#cv2.imshow('Edged1', edged1)

cv2.imwrite('./edged.bmp', edged)
cv2.imwrite('./edged1.bmp', edged1)

cv2.waitKey(0)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
closed1 = cv2.morphologyEx(edged1, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('closed', closed)
cv2.imwrite("closed1.bmp", closed1)
cv2.imwrite("closed.bmp", closed)

#cv2.waitKey(0)

## contour
contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
#cv2.imshow('contours_image', contours_image)
cv2.imwrite('contoured.bmp', contours_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

contours_xy = np.array(contours)
print(contours_xy.shape)

'''
x_min, x_max = 0, 0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
print(x_min)
print(x_max)

# y의 min과 max 찾기
y_min, y_max = 0, 0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)
print(y_min)
print(y_max)

# image trim 하기
x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min


img_trim = image[y:y+h, x:x+w]
# cv2.imwrite('trim.jpg', img_trim)
# org_image = cv2.imread('trim.jpg')
'''
