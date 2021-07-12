
import os
import numpy as np
from tensorflow.keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, \
    GlobalAveragePooling2D, Flatten
from tensorflow.keras import Input
import keras
import tensorflow as tf
import tensorflow.keras
import numpy as np
from PIL import Image
from keras.utils import np_utils
import math
from keras.optimizers import SGD

def _parse_tfrecord():
    def parse_tfrecord(tfrecord):
        features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                    'image/filename': tf.io.FixedLenFeature([], tf.string),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)

        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        y_train = tf.cast(x['image/source_id'], tf.int64)
        x_train = _transform_images()(x_train)
        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train
    return parse_tfrecord


def _transform_images():
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (224, 224))
        x_train = tf.image.random_crop(x_train, (224,224, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train

def load_tfrecord_dataset(tfrecord_name, batch_size, shuffle=True, buffer_size=1000):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset




## model layer
def inception(x_input, filter_1, filter_3_R, filter_3, filter_5_R, filter_5, pool_proj):
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='SAME')(x_input)
    x1 = Conv2D(filters=pool_proj, kernel_size=(1, 1), padding="SAME", activation='relu')(x1)

    # conv 1x1 -> conv 5x5 == conv 5x5 reduction -> conv 5x5
    x2 = Conv2D(filters=filter_5_R, kernel_size=(1, 1), padding="SAME", activation='relu')(x_input)
    x2 = Conv2D(filters=filter_5, kernel_size=(5, 5), padding="SAME", activation='relu')(x2)

    x3 = Conv2D(filters=filter_3_R, kernel_size=(1, 1), padding="SAME", activation='relu')(x_input)
    x3 = Conv2D(filters=filter_3, kernel_size=(3, 3), padding="SAME", activation='relu')(x3)

    x4 = Conv2D(filters=filter_1, kernel_size=(1, 1), padding="SAME", activation='relu')(x_input)

    # COncatenate each layers' depth
    inception_result = Concatenate()([x1, x2, x3, x4])

    return inception_result


###
def decay(epoch, steps=100):
    initial_lrate = 0.00001
    drop = 0.96
    epoch_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
    return lrate

def main():

    train_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train.tfrecord"
    val_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/val.tfrecord"

    train_dataset = load_tfrecord_dataset(train_url,128)
    val_dataset = load_tfrecord_dataset(val_url,128)

    ## INPUT = 244 X 244, RGB channel
    input_data = Input(shape=(224, 224, 3))

    ################# PART 1 ######################

    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME", activation='relu')(input_data)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation='relu')(x)

    x = tf.keras.layers.LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)

    ################# inception 3a, inception 3b ######################
    x = inception(x, 64, 96, 128, 16, 32, 32)
    x = inception(x, 128, 128, 192, 32, 96, 64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)

    ################# inception 4a ~ inception 4e ######################
    x = inception(x, 192, 96, 208, 16, 48, 64)

    # auxiliary classifier for over influencing
    ax1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    ax1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation='relu')(ax1)
    ax1 = Flatten()(ax1)
    ax1 = Dense(1024, activation="relu")(ax1)  ## FC
    ax1 = Dropout(0.7)(ax1)
    ax1 = Dense(1000, activation="softmax", name='ax1')(ax1)  ## FC

    x = inception(x, 160, 112, 224, 24, 64, 64)
    x = inception(x, 128, 128, 256, 24, 64, 64)
    x = inception(x, 112, 144, 288, 32, 64, 64)

    # auxiliary classifier for over influencing
    ax2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    ax2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation='relu')(ax2)
    ax2 = Flatten()(ax2)
    ax2 = Dense(1024, activation="relu")(ax2)  ## FC
    ax2 = Dropout(0.7)(ax2)
    ax2 = Dense(1000, activation="softmax", name='ax2')(ax2)  ## FC

    x = inception(x, 256, 160, 320, 32, 128, 128)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")(x)

    ################# inception 5a, inceptino 5b ######################
    x = inception(x, 256, 160, 320, 32, 128, 128)
    x = inception(x, 384, 192, 384, 48, 128, 128)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    ## add
    add = Dense(1000, activation='relu')(x)

    outputs = Dense(1000, activation="softmax", name='main_classifier')(add)

    # concat_output = tf.keras.layers.concatenate([outputs, ax1, ax2])

    model = tf.keras.models.Model(inputs=input_data, outputs=[outputs, ax1, ax2], name='googlenet')
  # model.summary()

    ###
    learning_rate = 0.1
    momentum = 0.9

    #optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    optimizer = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = SGD(momentum=0.9)
    # optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=False)
    # optimizer = keras.optimizers.Adam(lr=learning_rate, epsilon=None, decay=0.0, amsgrad=False)
    # optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = tf.keras.losses.kl_divergence
    loss = tf.keras.losses.CategoricalCrossentropy( from_logits=False, label_smoothing=0, reduction="auto", name="categorical_crossentropy", )



    model.compile(optimizer=optimizer,
                   loss={'main_classifier' : loss,
                         'ax1' : loss,
                         'ax2' : loss},
                   loss_weights={'main_classifier': 1.0,
                         'ax1': 0.3,
                         'ax2': 0.3},
                  metrics=['acc'])


    checkpoint_path = "checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # 모델의 가중치를 저장하는 콜백 만들기


    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
    ]

    val_count = 500 #len(val_dataset)
    train_count = 1000 # len(train_dataset)# len(train_dataset)


    BATCH_SIZE = 32
    steps_per_epoch = int(train_count/BATCH_SIZE)
    validation_steps = int(val_count/BATCH_SIZE)

    model.fit(train_dataset, epochs=50, batch_size=BATCH_SIZE,  steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    #모델 저장하기
    model.save('my_googLeNet.h5')


if __name__ == '__main__':
    main()









