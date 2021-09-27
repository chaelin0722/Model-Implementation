#############################################################################################
# https://www.kaggle.com/raufmomin/vision-transformer-vit-fine-tuning
#############################################################################################


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.preprocessing.image import ImageDataGenerator
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from vit_keras import vit, visualize

# warnings.filterwarnings('ignore')


IMAGE_SIZE = 224
BATCH_SIZE = 40
EPOCHS = 7

filenames =os.listdir("/home/ivpl-d28/PycharmProjects/NOAH/dataset/test_dataset/")
TRAIN_PATH = '/home/ivpl-d28/PycharmProjects/NOAH/dataset/test_dataset'

# TEST_IMAGES = glob.glob(TRAIN_PATH + '/*.jpg')

categories = []
for filename in filenames:
    category = filename.split('_')[0]
    if category == 'abnormal':
        categories.append(1)
    else:
        categories.append(0)

DF_TRAIN = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


DF_TRAIN['category'] = DF_TRAIN['category'].astype('str')


def data_augment(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if p_spatial > .75:
        image = tf.image.transpose(image)

    # Rotates
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    # Pixel-level transforms
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)

    return image

datagen = ImageDataGenerator(rescale = 1./255,
                              samplewise_center = True,
                              samplewise_std_normalization = True,
                              validation_split = 0.1,
                             )
                              #preprocessing_function = data_augment)

train_gen = datagen.flow_from_dataframe(dataframe = DF_TRAIN,
                                        directory = TRAIN_PATH,
                                        x_col="filename",
                                        y_col="category",
                                        target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                        subset ="training",
                                        batch_size = 40,
                                        seed = 1,
                                        color_mode = 'rgb',
                                        shuffle=True,
                                        class_mode="binary")

valid_gen = datagen.flow_from_dataframe(dataframe = DF_TRAIN,
                                        directory = TRAIN_PATH,
                                        x_col="filename",
                                        y_col="category",
                                        subset = "validation",
                                        batch_size =40,
                                        # seed = 1,
                                        color_mode = 'rgb',
                                        shuffle = False,
                                        class_mode = "binary",
                                        target_size = (IMAGE_SIZE, IMAGE_SIZE))



vit_model = vit.vit_b32(
        image_size = IMAGE_SIZE,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 2)


model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(11, activation = tfa.activations.gelu), #tf.keras.layers.ReLU),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(5, 'softmax') #          tf.keras.layers.Dense(5, 'softmax')
    ],
    name = 'vision_transformer')

model.summary()

## train

learning_rate = 1e-4

optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate) ## since installing tf_addons have problems....
# optimizer = tf.optimizers.Adam(learning_rate = learning_rate)

model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])


#STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
#STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size
#print(f'stepsizetrain : {STEP_SIZE_TRAIN}')


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',
                                                 factor = 0.2,
                                                 patience = 2,
                                                 verbose = 1,
                                                 min_delta = 1e-4,
                                                 min_lr = 1e-6,
                                                 mode = 'max')

earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                                                 min_delta = 1e-4,
                                                 patience = 5,
                                                 mode = 'max',
                                                 restore_best_weights = True,
                                                 verbose = 1)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = '/home/ivpl-d28/PycharmProjects/NOAH/dataset/vit_checkpoints/vit_best_weights.h5',
                                                  monitor = 'val_accuracy',
                                                  verbose = 1,
                                                  save_best_only = True,
                                                  save_weights_only = True,
                                                  mode = 'max')

callbacks = [earlystopping, reduce_lr, checkpointer]


#steps per epoch =  len(train_data)/batch_size
steps_per_epochs = int(len(DF_TRAIN)/BATCH_SIZE)

history = model.fit(x = train_gen,
                     steps_per_epoch = steps_per_epochs,
                     validation_data = valid_gen,
                     validation_steps = steps_per_epochs,
                     epochs = 20)
                     #callbacks = checkpointer)

model.save('./vit_model.h5')
