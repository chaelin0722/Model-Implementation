
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import tensorflow as tf
from efficientnet.keras import EfficientNetB3
from keras.models import load_model
from sklearn.metrics import confusion_matrix


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
from PIL import Image

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR = "./logs"
DEFAULT_IMAGE_DIR = "./val"
DEFAULT_MRCNN_MODEL_DIR = "./mask_rcnn_toothbrush_head_0015.h5"
DEFAULT_EFF_MODEL_DIR = '/home/clkim/PycharmProjects/NOAH/dataset/checkpoints/efficient-best_weight_1014.h5'

############################################################
#  Configurations
############################################################


class ToothBrushHeadConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "toothbrush_head"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, img_file_name=None):
    assert image_path


    # Run model detection and generate the color splash effect
    start_inference = time.time()
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)

    # Detect objects

    r = model.detect([image], verbose=1)[0]

    ## check time for inference
    end_inference = time.time()
    inference_time = end_inference - start_inference
    print(f"{inference_time:.2f} sec for inferencing {img_file_name}")


    # bounding box visualize
    class_names = ['background', 'defect']
    bbox = utils.extract_bboxes(r['masks'])
    file_name_bb = "bb_splash_{}".format(img_file_name)
    save_path_bb = os.path.join(DEFAULT_IMAGE_DIR, 'result', file_name_bb)

    # print("image_path", image_path)

    visualize.display_instances(save_path_bb, image_path, image, bbox, r['masks'], r['class_ids'], class_names,
                                r['scores'])


    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{}".format(img_file_name)
    save_path = os.path.join(DEFAULT_IMAGE_DIR , 'result', file_name)
    skimage.io.imsave(save_path, splash)

    print("Saved to ", save_path)

    ##



############################################################
#  classification
############################################################
def binary_classification(imgname, model):


    test_dir = os.path.join(DEFAULT_IMAGE_DIR+'/cropped/'+imgname)


    test_datagen = ImageDataGenerator(
        rescale=1 / 255
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(32, 32),
        batch_size=1,
        shuffle=False,
        class_mode=None
    )

    preds = model.predict_generator(test_generator, steps=len(test_generator.filenames))

    print(preds)
    image_ids = [name.split('/')[-1] for name in test_generator.filenames]
    predictions = preds.flatten()

    error = []
    for i in range(len(test_generator.filenames)):
        if predictions[i] > 0.5:
            error.append('error')

        else:
            error.append('normal')

    data = {'filename': image_ids, 'true_label': test_generator.classes, 'category': error}

    submission = pd.DataFrame(data)
    # final classification! whether error or not
    result = []
    if (submission['category'] == 'error').any():
        print(f'{imgname} is error tooth brush')
        result.append(imgname)
        #print("error toothbrush")

    return result
    #submission.to_csv("./test_result.csv", index=False)


############################################################
#  main
############################################################

if __name__ == '__main__':

    class InferenceConfig(ToothBrushHeadConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = InferenceConfig()

    config.display()
    #####

    result_path = DEFAULT_IMAGE_DIR+'/result'
    cropped_path = DEFAULT_IMAGE_DIR+'/cropped'

    if not os.path.isdir(result_path):
        os.mkdir(DEFAULT_IMAGE_DIR+'/result')
        os.mkdir(DEFAULT_IMAGE_DIR+'/cropped')


################# LOAD MODELS #######################
    #### MASK-R-CNN
    mrcnn_model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    weights_path = DEFAULT_MRCNN_MODEL_DIR
    mrcnn_model.load_weights(weights_path, by_name=True)

    ##### EFFICIENTNET

    efficient_net = EfficientNetB3(
        weights='imagenet',
        input_shape=(32, 32, 3),
        include_top=False,
        pooling='max'
    )

    eff_model = Sequential()
    eff_model.add(efficient_net)
    eff_model.add(Dense(units=120, activation='relu'))
    eff_model.add(Dense(units=120, activation='relu'))
    eff_model.add(Dense(units=1, activation='sigmoid'))
    eff_model.summary()

    eff_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    eff_model.load_weights(DEFAULT_EFF_MODEL_DIR)
    #############################################################################3

    # each image in folder
    image_path = DEFAULT_IMAGE_DIR
    dirs = os.listdir(image_path)
    # print(dirs)
    images = [file for file in dirs if file.endswith('.bmp')]
    # print("len iamges :::: ", len(images))
    for img in images:
        imgname = os.path.join(image_path, img)
        onlyname, _ = os.path.splitext(img)
        imgname_png = onlyname + '.png'
        # output_imgname = os.path.join(image_path, imgname_png)
        detect_and_color_splash(mrcnn_model, image_path=imgname, img_file_name=imgname_png)

        err_toothbrush_list = binary_classification(onlyname, eff_model)
        print(err_toothbrush_list)