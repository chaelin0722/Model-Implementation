from PIL import Image
import os, glob
import numpy as np
from keras.models import load_model
import keras

image_dir = "/home/ivpl-d14/PycharmProjects/imagenet/imagenet/test_4classes"
image_w = 244
image_h = 244

X = []
filenames = []
files = glob.glob(image_dir+"/*.JPEG")
for i, f in enumerate(files):
    img = Image.open(f)
    # img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)
model = load_model('my_googLeNet.h5')

model.load_weights('./checkpoints/checkpoint-epoch-100-batch-64-trial-002.h5')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
cnt = 0


for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블
    print("i : ",i)
    print("pre_ans: ", pre_ans)
    pre_ans_str = ''
    if pre_ans == 270: pre_ans_str = "wood_rabbit"
    elif pre_ans == 954: pre_ans_str = "warplane"
    elif pre_ans == 284: pre_ans_str = "wolf_spider"
    elif pre_ans == 511: pre_ans_str = "african_chameleon"
    else : print("cannot find")


    if i[0] >= 0.8 : print(filenames[cnt].split("/")[1]+"image is predicted as "+pre_ans_str)
    if i[1] >= 0.8 : print(filenames[cnt].split("/")[1]+"image is predicted as "+pre_ans_str)
    if i[2] >= 0.8 : print(filenames[cnt].split("/")[1]+"image is predicted as "+pre_ans_str)
    if i[3] >= 0.8 : print(filenames[cnt].split("/")[1]+"image is predicted as "+pre_ans_str)
    cnt += 1

