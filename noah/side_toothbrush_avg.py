import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
import time

def getMinMax(c_image):
    ## contour
    contours, _ = cv2.findContours(c_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_xy = np.array(contours)

    # min max of x
    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)

    print("x_min : ",x_min)
    print("x_max : ",x_max)

    # min max of y
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)

    print("y_min : ", y_min)
    print("y_max : ", y_max)

    # definition of x, y, w, h
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    return x, y, w, h

def preprocess(input):
    edged = cv2.Canny(input, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return closed

if __name__ == '__main__':

    image_name = 'pic_thre_1_black.bmp'
    image = cv2.imread(image_name )

    start = time.time()

    blur = cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    closed = preprocess(blur)

    # get minmax for whole image
    x, y, w, h = getMinMax(closed)

    ## trim image to check brushes' minmax
    #img_trim = image[y:y+h, x:x+w]
    img_trim_30 = image[y:y+h, x:x+30]
    #cv2.imwrite('trim_30.jpg', img_trim_30)


    closed2 = preprocess(img_trim_30)

    ## get minmax for only the brush
    x, y, w, h= getMinMax(closed2)

    ## get exact height of the brush and width for 10! to lessen calculation
    n = int(h/10)
    sum_wpix = []
    for i in range(10):
        height = y+(n*i)
        each_img = img_trim_30[height:height + n, x:x + 10]
        #check
        cv2.imwrite(f'trimmed_{i}.jpg', each_img)

        # counting the number of pixels
        num_wpix = np.sum(each_img == 255)
        sum_wpix.append(num_wpix)
        # num_bpix = np.sum(each_img == 0)
        print(f"{i}_num_wpix : ",num_wpix)

    sum = sum(sum_wpix)
    avg = sum / len(sum_wpix)
    print(sum_wpix)
    print("average : ", avg)

    new = []
    new = sum_wpix - avg
    print(new)

    err = []
    for num in new:
        if num >=50:
            print("error brush")
            err.append(num)

    ## error toothbrush
    if len(err) >= 2:
        print(f'{image_name} is error toothbrush')

    end = time.time()
    total_time = end-start
    print("ttime = ",total_time)
'''
    ## detect error toothbrush each with average
    err = []
    for pix in sum_wpix:
        print(pix)
        if pix >= avg:
            print("error brush")
            err.append(pix)

    print("err # : ", len(err))
    ## error toothbrush
    if len(err) >= 2:
        print(f'{image_name} is error toothbrush')

    end = time.time()
    total_time = end-start
    print("ttime = ",total_time)
'''
