import glob
import os
import cv2
import numpy as np
img_dir = "~/test/masks/"
gsd = (50 / 18.9) / 100 * (50 / 18.9) / 100
print(gsd)
# con = 0.0049
dirs = os.listdir(img_dir)
images = [file for file in dirs if file.endswith('.png') or file.endswith('.JPG')]
print(images)
for item in images:
    item_path = os.path.join(img_dir, item)
    mask = cv2.imread(item_path, cv2.IMREAD_GRAYSCALE)
    #print("maskkkkk : ", mask == 3)
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_0 = (mask == 0).astype(np.uint8)
    mask_1 = (mask == 1).astype(np.uint8)
    mask_2 = (mask == 2).astype(np.uint8)
    mask_3 = (mask == 3).astype(np.uint8)
    count_0 = np.count_nonzero(mask_0)
    count_1 = np.count_nonzero(mask_1)
    count_2 = np.count_nonzero(mask_2)
    count_3 = np.count_nonzero(mask_3)
    print(f"{item.split('/')[-1]},{count_1 * gsd},{count_2 * gsd},{count_3 * gsd}")


