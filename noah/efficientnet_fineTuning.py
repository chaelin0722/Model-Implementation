import numpy as np # linear algebra
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras import applications
from efficientnet.keras import EfficientNetB3
from keras import callbacks
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import glob, warnings



filenames = os.listdir("./test_dataset/")
train_dir = './test_dataset'


TEST_IMAGES = glob.glob(train_dir + '/*.jpg')

print(TEST_IMAGES)
categories = []
for filename in filenames:
    category = filename.split('_')[0]
    if category == 'abnormal':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

'''
normal = []

for i in range(len(filenames)):
    normal.append(cv2.imread(train_dir + '/' + df['filename'][i]))


labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

plt.figure(figsize=[10,10])
for x in range(0,6):
    plt.subplot(3, 3, x+1)
    plt.imshow(normal[x])
    plt.title(labels[x])
    x += 1

#plt.show()
'''

df['category'] = df['category'].astype('str')
print(df)

train_datagen = ImageDataGenerator(
    rescale=1/255,
   # validation_split=0.10,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("df : ",df)
train_generator = train_datagen.flow_from_dataframe(
    dataframe = df,
    directory = train_dir,
    x_col="filename",
    y_col="category",
    target_size=(32,32),
    subset="training",
    batch_size=1024,
    shuffle=True,
    class_mode="binary"
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe = df,
    directory = train_dir,
    x_col="filename",
    y_col="category",
    target_size=(32,32),
    subset="validation",
    batch_size=256,
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

history = model.fit_generator(
    train_generator,
    epochs = 1,
    steps_per_epoch = 15,
    validation_data = val_generator,
    validation_steps = 7
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
