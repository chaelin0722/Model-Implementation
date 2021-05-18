
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

batch_size = 128  #arbitary//
num_classes = 10
epochs = 50

#input image 32x32, each characters should center aligned. each pixels white-> -1.0, black -> 1.175
img_rows, img_cols = 32, 32

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential([
    # C1 (first layer)
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(32,32,1)),

    # S2 subsampling    # strides=(2,2) it means no overlapping
    tf.keras.layers.AveragePooling2D( pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None),
    # C3
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(28,28,6) ),
    # S4
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None,),
    # Flatten
    tf.keras.layers.Flatten(),
    # C5
    tf.keras.layers.Dense(120, activation='relu'),
    # F6
    tf.keras.layers.Dense(84, activation='relu'),
    # outputlayer, Softmax는 마지막 레이어의 결과값을 다중분류를 위한 확률값으로 해석할 수 있도록 하기 위함
    tf.keras.layers.Dense(10, activation='softmax')
])
## compile
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer="adam",
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##-- summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()