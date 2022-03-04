import cv2, os
import numpy as np
import time

# each image in folder
#image_path = '/home/ivpl-d28/Pycharmprojects/NOAH/dataset/side_dataset/'
image_path = '/home/ivpl-d28/Pycharmprojects/NOAH/dataset/side_dataset/300_/'
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

    # cv2.imshow('result_image', image)
    # cv2.waitKey(0)

    y, x= img.shape[:2]

    # right image trim
    img = img[:, :x-40]
    img1 = img1[:, :x-40]
    h, w = img.shape[:2]
    h1, w1 = img1.shape[:2]

    w_trim = np.sum(img[:10, :10] == 255)

    if w_trim >= 5:
        img = img[40:, :]
        img1 = img1[40:, :]
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


    roi_start = time.time()
    cv2.rectangle(img1, (x_min - 18, y_min - 23), (x_max + 18, y_max), (40,40,40), -1)

    roi_end = time.time()
    print("do rect : ", roi_end - roi_start)
    # result area
    roi_img = img1[:y + h - 10, :]

    cv2.imwrite(f'/home/ivpl-d28/Pycharmprojects/NOAH/dataset/side_dataset/300_/roi{_img}', roi_img)
    #make binary image
    roi_gray_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(roi_gray_img, 90, 255, cv2.THRESH_BINARY)

    #cv2.imshow("binary", thresh1)
    #cv2.waitKey(0)
    a, b, c = roi_img.shape
    pix_start = time.time()
    num_wpix = np.sum(thresh1 == 255)

    pix_end = time.time()

    print("pix count time : ", pix_end - pix_start)

    if num_wpix <= 10:
        norm_list.append(_img)

    total_end = time.time()
    print("total count time : ", total_end - total_start)
    #cv2.imwrite(f'/home/ivpl-d28/Pycharmprojects/NOAH/dataset/side_dataset/result/{_img}', thresh1)
    # cv2.imshow('result_image', roi_img)
    # cv2.waitKey(0)
    '''
    canny = cv2.Canny(roi_img, 90, 255)
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 80, minLineLength = 10, maxLineGap = 10)
    cv2.imshow('canny_image', canny)
    #cv2.imwrite('result_image.bmp', canny)
    cv2.waitKey(0)

    count = 0

    for line in lines:
        x1,y1,x2,y2 = line[0]
        theta = line[0][1]
        #if theta > 90 and theta < 300:
        cv2.line(roi_img,(x1,y1),(x2,y2),(0,255,0),1)
        count += 1

    print('the number of lines: ', count)

    #cv2.imshow('result_image', roi_img)
    #cv2.imwrite('result_image.bmp', roi_img)
    #cv2.waitKey(0)
    '''


print(norm_list)
