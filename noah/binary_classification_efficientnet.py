import numpy as np # linear algebra
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras import applications
from efficientnet.keras import EfficientNetB3
import keras
from keras import callbacks
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import datetime

#위 행(line)은 runs/logs 폴더를 생성합니다.

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"



data_filenames = os.listdir("./arbitrary_labeled_dataset/")
data_dir = './arbitrary_labeled_dataset/'



categories = []
for filename in data_filenames:
    category = filename.split('_')[0]
    if category == 'error':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': data_filenames,
    'category': categories
})

df['category'] = df['category'].astype('str')



datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.10,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# print("df : ",df)
train_generator = datagen.flow_from_dataframe(
    dataframe = df,
    directory = data_dir,
    x_col="filename",
    y_col="category",
    target_size=(32,32),
    subset="training",
    batch_size=128,
    shuffle=True,
    class_mode="binary"
)

val_generator = datagen.flow_from_dataframe(
    dataframe = df,
    directory = data_dir,
    x_col="filename",
    y_col="category",
    target_size=(32,32),
    subset="validation",
    batch_size=64,
    shuffle=True,
    class_mode="binary"
)

efficient_net = EfficientNetB3(
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

filename = './checkpoints/efficient-weights.h5'
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    # 개선된 validation score를 도출해낼 때마다 weight를 중간 저장
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filename,
                    save_weights_only=True,
                    verbose=1,  # 로그를 출력
#                    save_best_only=True,  # 가장 best 값만 저장
                    mode='auto')  # auto는 알아서 best를 찾습니다. min/max


steps_per_epochs = int((len(df) * 0.9) // train_generator.batch_size)
step_size_valid = int((len(df) * 0.1) // val_generator.batch_size)

#print(steps_per_epochs)  # 18
#print(step_size_valid)   # 4

history = model.fit_generator(
    train_generator,
    epochs = 50,
    steps_per_epoch = steps_per_epochs,
    validation_data = val_generator,
    validation_steps = 7,
    callbacks = [cp_callback],
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc,'b',label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


model.save_weights("./best_weight.h5")
