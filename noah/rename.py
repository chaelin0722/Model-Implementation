import os
import cv2

dir = "./validation_crop/"

'''
for folders in os.listdir(dir):
    src = os.path.join(dir, folders)
    dst = "./dataset/label_val_crop/"
    for files in os.listdir(src):
        name = str(os.path.splitext(files)[0]) + "_" + folders + ".jpg"
        ddir = os.path.join(folders,files)
        os.rename(os.path.join(dir, ddir), dst + name)

        #cv2.imwrite(dst, image)

'''

print(len(os.listdir('/home/clkim/PycharmProjects/NOAH/dataset/label_val_crop')))

print(len(os.listdir('/home/clkim/PycharmProjects/NOAH/dataset/label_train_crop')))
