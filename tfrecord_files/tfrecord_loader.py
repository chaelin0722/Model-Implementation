import tensorflow as tf


def parse_tfrecord(tfrecord):
    features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)}
    x = tf.io.parse_single_example(tfrecord, features)

    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    y_train = tf.cast(x['image/source_id'], tf.int64)
    x_train = transform_images(x_train)
    y_train = _transform_targets(y_train)

    return x_train, y_train


def parse_tfrecord_no_transform(tfrecord):

    features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)}
    x = tf.io.parse_single_example(tfrecord, features)

    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    y_train = tf.cast(x['image/source_id'], tf.int64)

    return x_train, y_train


def transform_images(x_train):
    x_train = tf.image.resize(x_train, (224, 224))
    x_train = tf.image.random_crop(x_train, (224,224, 3))
    x_train = tf.image.random_flip_left_right(x_train)
    x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
    x_train = tf.image.random_brightness(x_train, 0.4)
    x_train = x_train / 255
    return x_train


def _transform_targets(y_train):
    return y_train

def load_tfrecord_dataset(tfrecord_name, batch_size, shuffle):

    """load dataset from tfrecord"""
    raw_dataset = tf.data.Dataset.list_files(tfrecord_name, seed=42)
    raw_dataset = tf.data.TFRecordDataset(raw_dataset)

    raw_dataset = raw_dataset.interleave(tf.data.TFRecordDataset,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)

    raw_dataset = raw_dataset.repeat()


    if shuffle is True:
        dataset = raw_dataset.map(
            parse_tfrecord,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    elif shuffle is False:
        dataset = raw_dataset.map(
            parse_tfrecord_no_transform,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10)


    return dataset



train_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/test2.tfrecord"
# val_url = "/home/ivpl-d14/PycharmProjects/pythonProject/model_implementation/Model-Implementation/tfrecords/tf_train/val.tfrecord"

train_dataset = load_tfrecord_dataset(train_url, 64, shuffle=True)
# val_dataset = load_tfrecord_dataset(val_url, 64, shuffle=False)
