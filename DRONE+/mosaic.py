import random

import cv2
import os
import glob
import numpy as np
from PIL import Image


def mosaic(img_dir, msk_dir, all_img_list, all_masks, idxs, output_size, scale_range, filter_scale=0.):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    output_mask = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        m_path = all_masks[idx]
        img = cv2.imread(os.path.join(img_dir, path))
        msk = cv2.imread(os.path.join(msk_dir, m_path))

        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            msk = cv2.resize(msk, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            output_mask[:divid_point_y, :divid_point_x, :] = msk

        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
            msk = cv2.resize(msk, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            output_mask[:divid_point_y, divid_point_x:output_size[1], :] = msk

        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            msk = cv2.resize(msk, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            output_mask[divid_point_y:output_size[0], :divid_point_x, :] = msk

        else:  # bottom-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            msk = cv2.resize(msk, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
            output_mask[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = msk

    return output_img, output_mask

img_dir = "/home/clkim/PycharmProjects/DRONE+/GeoSeg/results/uavid_color/images/"
msk_dir = "/home/clkim/PycharmProjects/DRONE+/GeoSeg/results/uavid_color/masks/"

img_dirs = os.listdir(img_dir)
all_img_list = [file for file in img_dirs if file.endswith('.png') or file.endswith('.JPG')]

msk_dirs = os.listdir(msk_dir)
all_masks= [file for file in msk_dirs if file.endswith('.png') or file.endswith('.JPG')]

output_size = (1600,1300)
idxs = random.sample(range(len(msk_dirs)), 4)
SCALE_RANGE = (0.3, 0.7)
FILTER_TINY_SCALE = 1 / 50

print("all_img_list", all_img_list)
print("all_masks", all_masks)

img, mask = mosaic(img_dir, msk_dir, all_img_list, all_masks, idxs, output_size, SCALE_RANGE, FILTER_TINY_SCALE)

cv2.imwrite("./mosaic_image.jpg", img)
cv2.imwrite("./mosaic_mask.jpg", mask)
