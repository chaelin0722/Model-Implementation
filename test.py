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


BATCH_SIZE = 32

import tensorflow as tf


# tfrecord decode
def parse_image(record):
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, features)
    image = tf.io.decode_jpeg(parsed_record['image_raw'], channels=3)
    label= tf.cast(parsed_record['label'], tf.int32)

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

# train dataloader (augmentation)
def get_dataset_train(path, batch_size=BATCH_SIZE):
    record_files = tf.data.Dataset.list_files(path, seed=42)    # seed: shuffle
    dataset = tf.data.TFRecordDataset(filenames=record_files, compression_type="GZIP")
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Data Agumentation
    dataset = dataset.map(lambda image, label : (tf.image.random_flip_left_right(image), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)             # flip
    dataset = dataset.map(lambda image, label: (tf.image.random_crop(image, size=[224, 224, 3]), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)     # crop
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10)

    return dataset

# val or test dataloader (no augmentation)
def get_dataset_val(path, batch_size=BATCH_SIZE):
    record_files = tf.data.Dataset.list_files(path, seed=42)
    dataset = tf.data.TFRecordDataset(filenames=record_files, compression_type="GZIP")
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda image, label: (tf.image.resize_with_pad(image, 224, 224), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset=dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

# tfrecord data load
train_dataset= get_dataset_train("/home/hjpark/pycharmProject/Alexnet/imagenet_prep/tf_records/train/*.tfrecord")
val_dataset= get_dataset_val("/home/hjpark/pycharmProject/Alexnet/imagenet_prep/tf_records/val/*.tfrecord")