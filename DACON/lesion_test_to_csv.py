import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from pathlib import Path
import base64
import cv2
import matplotlib.pyplot as plt
import json
from tqdm.notebook import tqdm
import scipy
#from joblib import Parallel , delayed
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.copy()
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

total_list = []
results = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
}

result_path = Path('/home/clkim/PycharmProjects/DACON/YOLOR/yolor/runs/test/yolor_test2')
#result_img = list(result_path.glob('*.png'))
result_label = list(result_path.glob('labels/*.txt'))

for i in result_label:

    with open(str(i),'r') as f:
        file_name = i.name.replace('.txt','.json')
        img_name = file_name.replace('.json','.jpg')
        test_img_path = '/home/clkim/PycharmProjects/DACON/dataset/images/test'
        ow,oh,_ = cv2.imread(os.path.join(test_img_path, img_name))[:,:,::-1].shape
        for line in f.readlines():
            corrdi = line[:-1].split(' ')
            label,xc,yc,w,h,score = corrdi
            if float(score) > 0.25:
                xc,yc,w,h,score = list(map(float,[xc,yc,w,h,score]))
                xc,w = np.array([xc,w]) * ow
                yc,h = np.array([yc,h]) * oh

                refine_cordi = xywh2xyxy([xc,yc,w,h])
                refine_cordi = np.array(refine_cordi).astype(int)
                x_min,y_min,x_max,y_max = refine_cordi

                results['file_name'].append(file_name)
                results['class_id'].append(label)
                results['confidence'].append(score)
                results['point1_x'].append(x_min)
                results['point1_y'].append(y_min)
                results['point2_x'].append(x_max)
                results['point2_y'].append(y_min)
                results['point3_x'].append(x_max)
                results['point3_y'].append(y_max)
                results['point4_x'].append(x_min)
                results['point4_y'].append(y_max)


df = pd.DataFrame(results)
df['class_id'] = df['class_id'].apply(lambda x:int(x))

pd.DataFrame(df).to_csv('./yolor_best_weight_epoch_10.csv', index = False)
