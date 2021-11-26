import numpy as np # linear algebra
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras import applications
from efficientnet.keras import EfficientNetB0
import keras
from keras import callbacks
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import datetime
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


train_data_dir = '/home/clkim/PycharmProjects/NOAH/dataset/label_train_crop/'
val_data_dir = '/home/clkim/PycharmProjects/NOAH/dataset/label_val_crop/'
data_filenames_train = os.listdir(train_data_dir)
data_filenames_val = os.listdir(val_data_dir)

categories_train = []
categories_val = []

# label binary with filename
for filename in data_filenames_train:
    category = filename.split('_')[0]
    if category == 'error':
        categories_train.append(1)
    else:
        categories_train.append(0)

for filename in data_filenames_val:
    category = filename.split('_')[0]
    if category == 'error':
        categories_val.append(1)
    else:
        categories_val.append(0)


df_train = pd.DataFrame({
    'filename': data_filenames_train,
    'category': categories_train
})


df_val = pd.DataFrame({
    'filename': data_filenames_val,
    'category': categories_val
})

df_train['category'] = df_train['category'].astype('str')
df_val['category'] = df_val['category'].astype('str')



datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# print("df : ",df)
train_generator = datagen.flow_from_dataframe(
    dataframe = df_train,
    directory = train_data_dir,
    x_col="filename",
    y_col="category",
    target_size=(32,32),
    batch_size=128,
    shuffle=True,
    class_mode="binary"
)

val_generator = validation_datagen.flow_from_dataframe(
    dataframe = df_val,
    directory = val_data_dir,
    x_col="filename",
    y_col="category",
    target_size=(32,32),
    batch_size=1,
    shuffle=True,
    class_mode="binary"
)


efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(32,32,3),
    include_top=False,
    pooling='max'
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 120, activation='relu'))
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 1, activation='sigmoid'))
model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_path = './checkpoints/efficient-best_weight_eff0_new_label_1110.h5'
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callback_list = [
                    keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                        # 개선된 validation score를 도출해낼 때마다 weight를 중간 저장
                    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                save_weights_only=True,
                                                                verbose=1,  # 로그를 출력
                                                                save_best_only=True,  # 가장 best 값만 저장
                                                                mode='auto')  # auto는 알아서 best를 찾습니다. min/max

                ]

steps_per_epochs = int((len(df_train) * 0.9) // train_generator.batch_size)
step_size_valid = int((len(df_val) * 0.1) // val_generator.batch_size)  # 0.. so not use

#print(steps_per_epochs)  # 18
#print(step_size_valid)   # 4

model.fit_generator(
    train_generator,
    epochs = 50,
    steps_per_epoch = steps_per_epochs,
    validation_data = val_generator,
    validation_steps = 7,
    callbacks = callback_list
)

model.save("./eff0_new_label_1110.h5")

