## checklist

## file hwakjangja..  .jpg or .png or .bmp
## test_dir name!! check test_dir in visualize.py too!


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
from efficientnet.keras import EfficientNetB3, EfficientNetB0
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
from PIL import Image

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn_combine.config import Config
from mrcnn_combine import model as modellib, utils
from mrcnn_combine import visualize
from mrcnn_combine import c_visualize
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR = "./logs"
DEFAULT_IMAGE_DIR = "/dataset/0124_dataset"
DEFAULT_BRUSH_DIR = "/dataset/mask_rcnn_toothbrush_head_0020.h5" 
DEFAULT_EFF_MODEL_DIR = '/checkpoints/efficient-best_weight_220119_2.h5'
DEFAULT_CRACK_DIR='./mask_rcnn_toothbrush_crack__0036_tt47.h5'
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



class ToothBrushCrackConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "toothbrush_crack"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4 #1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7



############################################################
#  Dataset
############################################################

def color_splash(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

crack_detect_time = []
def crack_detect_and_color_splash(model, image_path=None, img_file_name=None):
    assert image_path
    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)

    start_inference = time.time()
    # Detect objects

    r = model.detect([image], verbose=1)[0]
    ## check time for inference
    end_inference = time.time()
    inference_time = end_inference - start_inference
    print(f"{inference_time:.2f} sec for inferencing_crack {img_file_name}")
    crack_detect_time.append(inference_time)



    # bounding box visualize
    class_names = ['bg','1','2','3','4']
    bbox = utils.extract_bboxes(r['masks'])
    file_name_bb = "bb_splash_{}".format(img_file_name)
    save_path_bb = os.path.join(DEFAULT_IMAGE_DIR, 'result_crack', file_name_bb)

    ## for check
    #print("class_ids", r['class_ids'])

    c_visualize.display_instances(save_path_bb, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
    # skimage.io.imsave(save_path_bb, bb_splash)
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    file_name = "splash_{}".format(img_file_name)
    save_path = os.path.join(DEFAULT_IMAGE_DIR, 'result_crack', file_name)
    skimage.io.imsave(save_path, splash)


    print("Saved to ", save_path)


brush_detect_time = []

def brush_detect_and_color_splash(model, image_path=None, img_file_name=None):
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
    print(f"{inference_time:.2f} sec for inferencing_brush {img_file_name}")
    brush_detect_time.append(inference_time)

    # bounding box visualize
    class_names = ['background', 'defect']
    bbox = utils.extract_bboxes(r['masks'])
    file_name_bb = "bb_splash_{}".format(img_file_name)
    save_path_bb = os.path.join(DEFAULT_IMAGE_DIR, 'result_brush', file_name_bb)

    # print("image_path", image_path)

    visualize.display_instances(DEFAULT_IMAGE_DIR, save_path_bb, image_path, image, bbox, r['masks'], r['class_ids'], class_names,
                                r['scores'])

#    print("scores =", r['scores'])
#    skimage.io.imsave(save_path_bb, bb_splash)

    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{}".format(img_file_name)
    save_path = os.path.join(DEFAULT_IMAGE_DIR , 'result_brush', file_name)
    skimage.io.imsave(save_path, splash)

    print("Saved to ", save_path)



############################################################
#  classification
############################################################
#result =[]
class_time = []

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
    start_classification = time.time()
    preds = model.predict_generator(test_generator, steps=len(test_generator.filenames))

    end_classification = time.time()

    ## check time
    inference_time = end_classification - start_classification
    print(f"{inference_time:.2f} sec for classification toothbrush hair")
    class_time.append(inference_time)


    # print(preds)
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

    if (submission['category'] == 'error').any():
        print(f'{imgname} is error tooth brush')
        #result.append(imgname)

        return imgname
    #else:
    #    return imgname, {'totally normal'}
############################################################
#  main
############################################################

if __name__ == '__main__':

    ## brush config
    class Brush_InferenceConfig(ToothBrushHeadConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    brush_config = Brush_InferenceConfig()

    brush_config.display()

    ##### crack config
    class Crack_InferenceConfig(ToothBrushCrackConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    crack_config = Crack_InferenceConfig()

    crack_config.display()
    #####



    brush_result_path = DEFAULT_IMAGE_DIR+'/result_brush'
    crack_result_path = DEFAULT_IMAGE_DIR+'/result_crack'
    cropped_path = DEFAULT_IMAGE_DIR+'/cropped'

    if not os.path.isdir(brush_result_path):
        os.mkdir(DEFAULT_IMAGE_DIR+'/result_brush')

    if not os.path.isdir(crack_result_path):
        os.mkdir(DEFAULT_IMAGE_DIR+'/result_crack')

    if not os.path.isdir(cropped_path):
        os.mkdir(DEFAULT_IMAGE_DIR+'/cropped')


    #### MASK-R-CNN brush
    brush_model = modellib.MaskRCNN(mode="inference", config=brush_config,
                                  model_dir=DEFAULT_LOGS_DIR)
    brush_weights_path = DEFAULT_BRUSH_DIR
    brush_model.load_weights(brush_weights_path, by_name=True)

    #### MASK-R-CNN crack
    crack_model = modellib.MaskRCNN(mode="inference", config=crack_config,
                                  model_dir=DEFAULT_LOGS_DIR)
    crack_weights_path = DEFAULT_CRACK_DIR
    crack_model.load_weights(crack_weights_path, by_name=True)

    ##### EFFICIENTNET
    efficient_net = EfficientNetB0(
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
    image_dir = os.path.join(image_path + "/test")
    dirs = os.listdir(image_dir)

    images = [file for file in dirs if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.bmp')]

    # print("len iamges :::: ", len(images))
    err_toothbrush_total = []
    each_toothbrush_info_total =[]
    submission = []
    ## arrange results and make csv file
    for img in images:
        imgname = os.path.join(image_dir, img)
        onlyname, _ = os.path.splitext(img)
        imgname_png = onlyname + '.png'
        # output_imgname = os.path.join(image_path, imgname_png)

        # CRACK
        crack_detect_and_color_splash(crack_model, image_path=imgname, img_file_name=imgname_png)

        # BRUSH
        brush_detect_and_color_splash(brush_model, image_path=imgname, img_file_name=imgname_png)

        brush_err_toothbrush = binary_classification(onlyname, eff_model)

        submission.append(brush_err_toothbrush)

        #print(each_toothbrush_preds)
        #print(err_toothbrush_list)
        #print("ddd_ time: ", brush_detect_time)
        #print("ccc_ time: ", class_time)
    with open("220119_result_info.csv", "wt",  encoding='utf-8-sig') as file:  #utf-8-sig for encode korean
        writer = csv.writer(file)
        writer.writerow(submission)


    ### compute time
    classi_avg = sum(class_time, 0.0) / len(class_time)
    brush_detect_avg = sum(brush_detect_time, 0.0) / len(brush_detect_time)
    crack_detect_avg = sum(crack_detect_time, 0.0) / len(crack_detect_time)

    print("average classification time : ", classi_avg)
    print("average Brush detection time : ", brush_detect_avg)
    print("average Crack detection time : ", crack_detect_avg)
    # print("average crack detection time : ", detect_avg)
