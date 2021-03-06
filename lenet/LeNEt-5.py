
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import datetime
import numpy as np
from PIL import Image

batch_size = 128  #arbitary//
num_classes = 10
epochs = 50

#input image 32x32, each characters should center aligned. each pixels white-> -1.0, black -> 1.175
img_rows, img_cols = 28,28  #change the size => add pixel using pad

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)  #grayscale image = 1, rgb = 3

print("inputshape",input_shape)


# Pad with 2 zeros on left and right hand sides-
padding = [[0, 0],
           [2, 2],
           [2, 2],
           [0, 0]]
# padding = [[Batch_size], [height], [width], [channel]]
x_train = np.pad(x_train[:,], padding, 'constant')
#  => (60000, 32, 32, 5)
x_test = np.pad(x_test[:,], padding, 'constant')
#  => (10000, 32, 32, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)  #x_train shape: (60000, 28, 28)
print(x_train.shape[0], 'train samples') #60000 train samples
print(x_test.shape[0], 'test samples') #10000 test samples

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

## for check
# plt.imshow(x_train[1], cmap=plt.cm.binary)
# plt.show()
# print(y_train[1])
# LeNet-5 model
# ==> need to add bias
model = tf.keras.models.Sequential([
    # C1 (first layer)
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), activation='tanh',input_shape=(32,32,1)),
    # S2 subsampling    # strides=(2,2) it means no overlapping
    tf.keras.layers.AveragePooling2D( pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None),
    # C3
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), activation='tanh', input_shape=(28,28,6)),
    # S4
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None),
    # Flatten
    tf.keras.layers.Flatten(),
    # C5
    tf.keras.layers.Dense(120, activation='tanh'),
    # F6
    tf.keras.layers.Dense(84, activation='tanh'),
    # outputlayer, Softmax??? ????????? ???????????? ???????????? ??????????????? ?????? ??????????????? ????????? ??? ????????? ?????? ??????
    tf.keras.layers.Dense(10, activation='softmax') #=> softmax for gaussian connection
])
## compile
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer="adam",
              metrics=['accuracy'])

# check tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

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

##-- summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## actual test for image
img = Image.open('../../deeplearning_seminar/mnist/dataset_test/testimgs/5.png').convert("L")
img = np.resize(img, (32,32,1))
im2arr = np.array(img)
im2arr = im2arr.astype('float32')
im2arr /= 255
im2arr = im2arr.reshape(1,32,32,1)
y_pred = model.predict_classes(im2arr)
print(y_pred)

# save model
#model.save('my_lenet_model.h5')
