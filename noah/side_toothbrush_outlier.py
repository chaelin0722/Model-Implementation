import cv2, sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import t

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


def generalized_esd(data, num_outliers, alpha=0.05):
    assert len(data) - num_outliers > 0, 'invalid num_outliers'
    n = len(data)
    temp_data = data.copy()
    res = []
    for i in range(num_outliers):
        mean = np.mean(temp_data)
        std = np.std(temp_data, ddof=1)
        diff = np.abs(temp_data - mean)
        R = np.max(diff) / std

        t_val = t.ppf(1 - alpha / (2 * (n - i)), n - i - 2)
        lambda_val = (n - i - 1) * t_val / np.sqrt((n - i - 2 + t_val ** 2) * (n - i))

        temp_idx = np.where(diff == np.max(diff))[0][0]
        temp_data_point = temp_data[temp_idx]
        idx = np.where(data == temp_data_point)[0][0]  ## index of suspected outlier
        value = data[idx]  ## suspected outlier
        flag = R > lambda_val
        res.append((idx, value, flag, R, lambda_val))
        temp_data = np.delete(temp_data, temp_idx)

    if res:
        idx_suspected_outlier = []
        for i, r in enumerate(res):
            if r[2] == True:
                idx_suspected_outlier.append(i)

        num_suspected_outlier = max(idx_suspected_outlier) + 1
        outlier_idx = [res[i][0] for i in range(num_suspected_outlier)]
        outlier_idx = np.array(outlier_idx)
        values = data[outlier_idx]
        Rs = [res[i][3] for i in range(num_suspected_outlier)]
        lambdas = [res[i][4] for i in range(num_suspected_outlier)]
        return (outlier_idx, values, Rs, lambdas)

    else:
        return False  ## no outlier detected



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
    #print(sum_wpix)
    print("average : ", avg)

    ## detect error toothbrush each with outlier
    sum_wpix = np.array(sum_wpix)

    res = generalized_esd(sum_wpix, num_outliers=3, alpha=0.05)

    ## check with plot
    if len(res) == 4:
        fig = plt.figure(figsize=(8, 8))
        fig.set_facecolor('white')
        plt.scatter(range(len(sum_wpix)), sum_wpix)
        plt.scatter(res[0], res[1], s=200, facecolor='none', edgecolors='r')
        plt.xticks(range(len(sum_wpix)), range(1, len(sum_wpix) + 1))
        plt.show()
    else:
        print(res)
        print('There is no outlier')



    end = time.time()
    total_time = end-start
    print("ttime = ",total_time)

