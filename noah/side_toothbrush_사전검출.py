
import cv2, os
import numpy as np
import time

image_path = 't'
dirs = os.listdir(image_path)
# print(dirs)
images = [file for file in dirs if file.endswith('.png') or file.endswith('.bmp')]
# print("len iamges :::: ", len(images))

norm_list = []
sum_wpix = []
w_trim=0
for _img in images:
    imgname = os.path.join(image_path, _img)

    total_start = time.time()
    image = cv2.imread(imgname)
    image = cv2.resize(image, (700, 500))
    img = image.copy()  # contour 좌표를 구하기 위한 원본 복사 이미지
    img1 = image.copy()  # ROI영역을 만들기 위한 원본 복사 이미지1
    img_morph = image.copy()
    # cv2.imshow('result_image', image)
    # cv2.waitKey(0)

    y, x= img.shape[:2]

    # right image trim
    img = img[:, :x-40]
    img1 = img1[:, :x-40]
    img_morph = img_morph[:, :x-40]
    h, w = img.shape[:2]
    h1, w1 = img1.shape[:2]

    w_trim = np.sum(img[:10, :10] == 255)

    if w_trim >= 5:
        img = img[40:, :]
        img1 = img1[40:, :]
        img_morph = img_morph[40:, :]
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
    #cv2.imshow('contours_image', contours_image)

    ##cv2.imwrite('contoured.bmp', contours_image)

    #cv2.waitKey(0)

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


    #roi_start = time.time()
    #cv2.rectangle(img1, (x_min - 18, y_min - 23), (x_max + 18, y_max), (40,40,40), -1)


    #roi_end = time.time()
    #print("do rect : ", roi_end - roi_start)
    # apply binary -> not roi image but whole image!!!
    morph_img = cv2.cvtColor(img_morph, cv2.COLOR_BGR2GRAY)
    ret, morph_thresh = cv2.threshold(morph_img, 90, 255, cv2.THRESH_BINARY)

    ## morphology
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # 열림 연산
    opening = cv2.morphologyEx(morph_thresh, cv2.MORPH_OPEN, open_k)

    cv2.rectangle(opening, (x_min - 18, y_min - 23), (x_max + 18, y_max), (0,0,0), -1)

    # result area
    roi_img = opening[:y + h - 10, :]
    thresh_img = morph_thresh[:y + h - 10, :]

    # 결과 출력
    merged = np.hstack((thresh_img, roi_img))

    # count pixels
    num_wpix = np.sum(roi_img == 255)


    # 이미지에 글자 합성하기
    result = cv2.putText(merged, f"{num_wpix} pixels!", (560, 230), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv2.LINE_AA)

    #cv2.imshow('Erode', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    if num_wpix <= 10:
        norm_list.append(_img)
        print(f"{_img} has {num_wpix} and it is normal")

    #total_end = time.time()
    #print("total count time : ", total_end - total_start)

    cv2.imwrite(f'ts/hstack_{_img}', result)

print(norm_list)
