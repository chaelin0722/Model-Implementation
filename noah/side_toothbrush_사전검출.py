import cv2, os
import numpy as np


# each image in folder
image_path = 'dataset/side_dataset/'
dirs = os.listdir(image_path)
# print(dirs)
images = [file for file in dirs if file.endswith('.png') or file.endswith('.bmp')]
# print("len iamges :::: ", len(images))

norm_list = []
sum_wpix = []
w_trim=0
for _img in images:
    imgname = os.path.join(image_path, _img)
    image = cv2.imread(imgname)
    image = cv2.resize(image, (700, 500))
    img = image.copy()  # contour 좌표를 구하기 위한 원본 복사 이미지
    img1 = image.copy()  # ROI영역을 만들기 위한 원본 복사 이미지1

    # cv2.imshow('result_image', image)
    # cv2.waitKey(0)

    h, w = img.shape[:2]
    h1, w1 = img1.shape[:2]

    w_trim = np.sum(img[:10, :10] == 255)

    if w_trim >= 5:
        img = image[40:, :]
        img1 = image[40:, :]
        h, w = img.shape[:2]
        h1, w1 = img1.shape[:2]

    for y in range(h):
        for x in range(w):
            if img[y, x][0] < 230 or img[y, x][1] < 230 or img[y, x][2] < 230:
                # y, x 순서인 이유 : 영상 행렬은 높이, 길이로 저장되므로
                img[y, x] = 0

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('result_image', gray_img)
    # cv2.waitKey(0)

    edged = cv2.Canny(gray_img, 10, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    ## contour
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0

    contours_image = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('contours_image', contours_image)

    ##cv2.imwrite('contoured.bmp', contours_image)

    # cv2.waitKey(0)

    contours_xy = np.array(contours)
    print(contours_xy.shape)

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
    w = x_max - x_min
    h = y_max - y_min



    for j in range(h1):
        for i in range(w1):
            if i >= x_min - 18 and i <= x_max + 18 and j >= y_min - 23 and j <= y_max:
                img1[j, i] = 40

    # result area
    roi_img = img1[:y + h - 5, :]

    #make binary image
    roi_gray_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(roi_gray_img, 127, 255, cv2.THRESH_BINARY)

    #cv2.imshow("binary", thresh1)
    #cv2.waitKey(0)
    a, b, c = roi_img.shape
    for j in range(a):
        for i in range(b):
            num_wpix = np.sum(thresh1 == 255)



    if num_wpix <= 10:
        norm_list.append(_img)
