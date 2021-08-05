import os
from tqdm import tqdm
from glob import glob
import random
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(dataset_path, output_path):
    samples = []
    print("Reading data list...")

    i = 1
    for image_name in tqdm(os.listdir(dataset_path)):
        img_paths = glob(os.path.join(dataset_path, image_name, '*.JPEG'))

        for img_path in img_paths:
          # filename = os.path.join(str(i), os.path.basename(img_path)) #filename: 1/n02102177_3388.JPEG
            id = str(i)
            samples.append((img_path, id))
        i = i+1
  #  print("sampleds :", samples)  # => ('/home/ivpl-d14/PycharmProjects/imagenet/imagenet/test/diaper/n03188531_11898.JPEG', '999'),
    random.shuffle(samples)
    print("Writing tfrecord file...")

    with tf.io.TFRecordWriter(output_path) as writer:
        for img_path, id in tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                      source_id=int(id))
        #    print("tf_example", tf_example)

            writer.write(tf_example.SerializeToString())

if __name__ == "__main__":
    dataset_path = "/home/ivpl-d14/PycharmProjects/imagenet/imagenet/train"
    output_path = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/train2.tfrecord"
    main(dataset_path, output_path)
