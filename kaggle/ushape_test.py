import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import pytorch_ssim
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from torch.nn.modules.loss import _Loss
from net.Ushape_Trans import *
#from dataset import prepare_data, Dataset
from net.utils import *
import cv2
import matplotlib.pyplot as plt
from utility import plots as plots, ptcolor as ptcolor, ptutils as ptutils, data as data
from loss.LAB import *
from loss.LCH import *
from torchvision.utils import save_image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


dtype = 'float32'

torch.set_default_tensor_type(torch.FloatTensor)


def split(img):
    output=[]
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output

# Initialize generator
generator = Generator().cuda()
generator.load_state_dict(torch.load("./saved_models/G/generator_795.pth"))


img_size = 256
path = '/home/clkim/Projects/kaggle/train_images/video_0'
path_gen = '/home/clkim/Projects/kaggle/train_images/video_0_gen'

def predict(img_paths, stride=128, batch_size=1):
    results = []

    for img_path in os.listdir(img_paths):
        if img_path in os.listdir(path_gen):
            print(f"{img_path}, already exist")
            continue
        else:
            print(f"{img_path}, is running")
            img = cv2.imread(os.path.join(img_paths, img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = np.array(img).astype(dtype)
            #img = torch.from_numpy(img)  # tensor (h, w, c)


            crop = []
            position = []
            batch_count = 0

            result_img = np.zeros_like(img)
            voting_mask = np.zeros_like(img)
            i = 0
            for top in tqdm(range(0, img.shape[0], stride)):
                for left in range(0, img.shape[1], stride):
                    piece = np.zeros([img_size, img_size, 3], np.float32)
                    temp = img[top:top + img_size, left:left + img_size, :]
                    piece[:temp.shape[0], :temp.shape[1], :] = temp
                    crop.append(piece)
                    position.append([top, left])
                    batch_count += 1
                    if batch_count == batch_size:
                        crop = np.array(crop).astype(dtype)
                        crop = torch.from_numpy(crop)  #tensor (b, h, w, c)
                        # permute() : 모든 차원의 순서를 재배치
                        crop = crop.permute(0, 3, 1, 2) # tensor (b, c, h, w)
                        #.unsqueeze(0) => make batch 1
                        crop = crop / 255.0
                        crop = Variable(crop).cuda()
                        output = generator(crop)*255
                        pred = output[3].data

                        # cropped image output
                        #save_image(pred, "./test/output/cropped_" + img_path, nrow=5, normalize=True)

                        crop = []
                        batch_count = 0
                        for num, (t, l) in enumerate(position):
                            piece = pred[num]
                            piece = piece.permute(1, 2, 0)
                            h, w, c = result_img[t:t + img_size, l:l + img_size, :].shape
                            #result_img = torch.Tensor(result_img).cuda()
                            piece = piece.cpu().detach().numpy()
                            result_img = result_img.astype(dtype)
                            result_img[t:t + img_size, l:l + img_size, :] += piece[:h, :w, :]
                            voting_mask[t:t + img_size, l:l + img_size, :] += 1
                        position = []

            result_img = result_img / voting_mask
            #result_img = result_img.astype(np.uint8)
            #results.append(result_img)
            results = np.array(result_img).astype(dtype)
            results = torch.from_numpy(results)
            results = results.permute(2, 0, 1).unsqueeze(0)# tensor (b, c, h, w)
            save_image(results, "/home/clkim/Projects/kaggle/train_images/video_0_gen/" + img_path, nrow=5, normalize=True)

        return results


if __name__ == "__main__":

    predict(path)














