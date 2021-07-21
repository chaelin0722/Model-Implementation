import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import cv2


dir = '/home/ivpl-d14/PycharmProjects/imagenet/imagenet/test/zucchini'
files = os.listdir(dir)
file_dir = []
image = []
image_arr = []
for i in range (4):
    single_file = random.choice(files)
    file_dir.append(os.path.join(dir, single_file))
    image = Image.open(file_dir[i])
    image_arr = np.array(image)

###

rows = 2
cols = 2

fig = plt.figure()


for num in range(4):  # num1 ~ num5
    i = 1
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(image_arr[i])
    ax.set_xlabel(num)
    ax.set_xticks([]), ax.set_yticks([])
    i+=1

fig.tight_layout()  # setting blanks between images
plt.show()




'''
dir = '/home/ivpl-d14/PycharmProjects/imagenet/imagenet/test/zucchini/*.JPEG'
##
def read_img(file_path):
    img_arr = cv2.imread(file_path)
    return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

img_arrs = []
img_num = range(0,50)

for i in random.sample(img_num, 4):
    img_arrs.append(read_img(dir[i]))

rows = 2
cols = 2

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*2, rows*2))

for num in range(1, rows*cols+1):  # num1 ~ num5
    fig.add_subplot(rows, cols, num)
    idx = num-1 # setting index

    plt.imshow(img_arrs[idx], aspect="auto")
    plt.xlabel(f'{img_arrs[idx].shape}', fontsize=12)

fig.tight_layout()  # setting blanks between images
'''

