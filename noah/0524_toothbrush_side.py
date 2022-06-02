import math

import cv2, os
import numpy as np
import time
'''

컨투어를 각 10개의 식모 모두에 대해서 하나씩 하면서 돌아가는 방법구현한 코드

..! -> error 가 많을 것 같아서 패스..ㅎㅎ

'''

image_path = ''
dirs = os.listdir(image_path)
# print(dirs)
images = [file for file in dirs if file.endswith('.png') or file.endswith('.bmp')]
# print("len iamges :::: ", len(images))

norm_list = []
pre_err_list = []
err_list_6 = []
err_list_65 = []
post_err_list = []
sum_wpix = []
w_trim = 0


def preprocessing(input):
    edged = cv2.Canny(input, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return closed


def getMinMax(image_c, t=0):
    ## contour
    contours, _ = cv2.findContours(image_c.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_xy = np.array(contours)
    # print(contours_xy.shape)

    if t == 1:
        contours_image = cv2.drawContours(image_c.copy(), contours, -1, (0, 0, 255), 3)

    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)
    # print(x_min)
    # print(x_max)

    # y의 min과 max 찾기
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)
    # print(y_min)
    # print(y_max)

    # image trim 하기
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    return x, y, w, h, x_min, x_max, y_min, y_max


def get_hole_distance(img):
    trim_handle = img[:y_max - 10, :]
    h_height, h_width = trim_handle.shape

    for f in range(h_width):
        if trim_handle[h_height - 1][f] == 255:
            print("f", f)
            fst_pix = f
            break

    for i in range(fst_pix, h_width - 1):
        if trim_handle[h_height - 1][i] == 0:
            fst_e_pix = i
            print("fst_e_pix", fst_e_pix)
            break

    for s in range(fst_e_pix, h_width - 1):
        if trim_handle[h_height - 1][s] == 255:
            print("sec", s)
            sec_pix = s
            break

    f_m = fst_pix + ((fst_e_pix - fst_pix) / 2)
    s_m = sec_pix + ((fst_e_pix - fst_pix) / 2)

    print(f_m)
    print(s_m)

    return (s_m - f_m)


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

    y, x = img.shape[:2]

    # right image trim
    img = img[:, :x - 40]
    img1 = img1[:, :x - 40]
    img_morph = img_morph[:, :x - 40]
    h, w = img.shape[:2]
    h1, w1 = img1.shape[:2]

    w_trim = np.sum(img[:10, :10] == 255)

    if w_trim >= 3:
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

    # preprocessing
    closed = preprocessing(gray_img)
    # cv2.imshow('result_image', gray_img)
    # cv2.waitKey(0)

    # get minmax
    x, y, w, h, x_min, x_max, y_min, y_max = getMinMax(closed)

    # apply binary -> not roi image but whole image!!!
    morph_img = cv2.cvtColor(img_morph, cv2.COLOR_BGR2GRAY)
    ret, morph_thresh = cv2.threshold(morph_img, 90, 255, cv2.THRESH_BINARY)

    ## morphology
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 열림 연산
    open = cv2.morphologyEx(morph_thresh, cv2.MORPH_OPEN, open_k)
    opening = open.copy()

    cv2.rectangle(opening, (x_min - 18, y_min - 23), (x_max + 18, y_max), (0, 0, 0), -1)

    # result area
    roi_img = opening[:y + h - 10, :]
    thresh_img = morph_thresh[:y + h - 10, :]

    # 결과 출력
    merged = np.hstack((thresh_img, roi_img))

    # count pixels
    num_wpix = np.sum(roi_img == 255)

    # 이미지에 글자 합성하기
    result = cv2.putText(merged, f"{num_wpix} pixels!", (560, 230), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1,
                         cv2.LINE_AA)

    if num_wpix > 18 and num_wpix <= 4000:
        # print(f'{_img} is error toothbrush')
        pre_err_list.append(_img)
    if num_wpix <= 18 or num_wpix > 4000:
        print(f'{_img} is checking - postprocessing')

        ####
        # check density 이부분을 새롭게 짬!
        ####
        '''
        trim_handle = morph_thresh[:y + h - 10, :]
        xx, yy, ww, hh, xx_min, xx_max, yy_min, yy_max = getMinMax(trim_handle)
        only_brushes = trim_handle[:, xx_min : xx_min + ww]

        _, org_w = only_brushes.shape
        n = int(org_w / 10)
        pok = []
        temp = 0
        for i in range(10):
            print(f"{i}, {temp}")
            each_brush_n = only_brushes[:, temp : temp + n]
            xx, yy, ww, hh, xx_min, xx_max, yy_min, yy_max = getMinMax(each_brush_n, 1)
            temp = xx_max + temp

            pok.append(ww)

        print("pok", pok)
        '''
            #cv2.imshow('dense_image', each_brush_n)
            #cv2.waitKey(0)

        # x, y, w, h, x_min, x_max, y_min, y_max

        #####
        # there are two algorithms
        # a. neighbor diff / hole_distance > 0.2
        # b. max-min / hole_distance > 0.4
        #####

        # 1. get hole to hole distance.
        hole_distance = get_hole_distance(morph_thresh)

        ## get minmax for only the brush
        x, y, w, h, x_min, x_max, y_min, y_max = getMinMax(open)
        img_trim_30 = morph_thresh[y_min: y_min + 30, x:x + w]

        # cv2.imshow("img_trim", img_trim_30)
        # cv2.waitKey(0)

        ## get minmax for trimmed image
        img_trim_30_open = img_trim_30.copy()
        img_trim_30_open = cv2.morphologyEx(img_trim_30_open, cv2.MORPH_OPEN, open_k)
        x, y, w, h, x_min, _, _, _ = getMinMax(img_trim_30_open)

        # cv2.imshow("img_trim_30_morph_open", img_trim_30)
        # cv2.waitKey(0)
        ## get exact width of the brush and width for 10! to lessen calculation
        n = int(w / 10)
        neighbor_diff = []
        min_pix = []
        for i in range(10):
            width = x_min + (n * i)
            # each_img = img_trim_30[height:height + n, x:x + 10]
            each_img = img_trim_30[:, width:width + n]

            # get the min of each trimmed image
            e_h, e_w = each_img.shape
            check_break = True
            for a in range(e_h):
                for b in range(e_w):
                    # print(a,",",b,"=",each_img[a][b])
                    if each_img[a][b] == 255:
                        min_pix.append(a)  # append height 좌표값
                        check_break = False
                        # print(a, ",", b, "=", each_img[a][b])
                        break
                    if a == e_h and b == e_w:
                        min_pix.append(30)

                if check_break == False:
                    break
            if i != 0:
                neighbor_diff.append(abs(min_pix[i] - min_pix[i - 1]))

        # a. neighbor diff / hole_distance > 0.2

        for i in range(len(neighbor_diff)):
            if round((neighbor_diff[i] / hole_distance), 2) > 0.2:
                post_err_list.append(_img)
                break

        # b. max-min / hole_distance > 0.4
        minmaxdiff = max(min_pix) - min(min_pix)
        if _img not in post_err_list and round((minmaxdiff / hole_distance), 2) > 0.4:
            post_err_list.append(_img)

        print("10 Min pixels : ", min_pix)
        print("neighbor diff : ", neighbor_diff)

    if _img not in pre_err_list and _img not in post_err_list:
        norm_list.append(_img)

    # total_end = time.time()
    # print("total count time : ", total_end - total_start)

print("preprocess err : ", pre_err_list)
print("postprocess err : ", post_err_list)  # <= 7
print("normal list : ", norm_list)
