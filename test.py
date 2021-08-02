import os


#charr = '898'

#toint = int(charr)
#print(i+toint)
#img_path = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train.tfrecord"

#filename = os.path.join(str(i), os.path.basename(img_path))

#a = 'python'

#print(a[:-1])

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

model.summary()