"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

results = {
    'file_name': [], 'class_id': [], 'confidence': [], 'point1_x': [], 'point1_y': [],
    'point2_x': [], 'point2_y': [], 'point3_x': [], 'point3_y': [], 'point4_x': [], 'point4_y': []
}

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR = "./logs"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


############################################################
#  Configurations
############################################################


class LeisonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "lesion"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

    print("DETECTION_CONFIDENCE", DETECTION_MIN_CONFIDENCE)
############################################################
#  Dataset
############################################################

class LesionDataset(utils.Dataset):

    def load_lesion(self, dataset_dir):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        '''
        self.add_class("lesion", 1, "01_ulcer")
        self.add_class("lesion", 2, "02_mass")
        self.add_class("lesion", 3, "04_lymph")
        self.add_class("lesion", 4, "05_bleeding")
        '''
        self.add_class("lesion", 1, "1")
        self.add_class("lesion", 2, "2")
        self.add_class("lesion", 3, "3")
        self.add_class("lesion", 4, "4")

        # Train or validation dataset?
        #assert subset in ["train_img"] #, "val"]
        dataset_dir = dataset_dir

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir,"via_region_data.json")))
        annotation_ = list(annotations)
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        i = 0
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                objects = [s['region_attributes']['category_id'] for s in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                objects = [s['region_attributes']['category_id'] for s in a['regions']]

            #name_dict = {"01_ulcer":1, "02_mass":2, "04_lymph":3, "05_bleeding":4}
            name_dict = {'1': 1, '2': 2, '3': 3, '4': 4}
            num_ids = [name_dict[a] for a in objects]

            print(f'image: {annotation_[i]}, num_ids : {num_ids}')
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, annotation_[i])# a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "lesion",
                image_id= annotation_[i], #a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids = num_ids)
            i += 1    

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "lesion":
            return super(self.__class__, self).load_mask(image_id)


        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "lesion":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids'] ## added
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # modify for rectangle
            #---------------------------------------------------------------------------------

            if p['name'] == 'rect':
                all_points_x = [p['x'], p['x'] + p['width'], p['x'] + p['width'], p['x']]
                all_points_y = [p['y'], p['y'], p['y'] + p['height'], p['y'] + p['height']]
                p['all_points_x'] = all_points_x
                p['all_points_y'] = all_points_y
            # ---------------------------------------------------------------------------------
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)
        print("load_ mask_num_ids", num_ids)
        return mask, num_ids
        # return mask.astype(np.bool_), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lesion":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LesionDataset()
    dataset_train.load_lesion(args.dataset)
    dataset_train.prepare()

    # Validation dataset
#    dataset_val = LesionDataset()
#    dataset_val.load_lesion(args.dataset, "val")
#    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, val_dataset = dataset_train,
                learning_rate=config.LEARNING_RATE,
                epochs=70,
                layers='3+')


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


def detect_and_color_splash(model, image_path=None, video_path=None, img_file_name=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]

        threshold = 0.5

        idx = np.where(r['scores'] > threshold)[0]

        for i in idx:
            x_min, y_min, x_max, y_max = r['rois'][i]
            class_id = r['class_ids'][i]
            print(f'class_id{i}', class_id)
            confidence = r['scores'][i]
            # think various bbox in an image!!!

            results['file_name'].append(img_file_name)
            results['class_id'].append(class_id)
            results['confidence'].append(confidence)
            results['point1_x'].append(x_min)
            results['point1_y'].append(y_min)
            results['point2_x'].append(x_max)
            results['point2_y'].append(y_min)
            results['point3_x'].append(x_max)
            results['point3_y'].append(y_max)
            results['point4_x'].append(x_min)
            results['point4_y'].append(y_max)
            print("class_id", class_id)


        # bounding box visualize
        class_names = ['background']
        for i in range(1, 5):
            class_names.append(str(i))
        ##3
        # class_names = ['background',1','2','3','4']

        bbox = utils.extract_bboxes(r['masks'])
        file_name_bb = "bb_splash_{}".format(img_file_name)
        save_path_bb = os.path.join(args.image, 'result', file_name_bb)
        visualize.display_instances(save_path_bb, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
        #skimage.io.imsave(save_path_bb, bb_splash)
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        file_name = "splash_{}".format(img_file_name)
        save_path = os.path.join(args.image, 'result', file_name)
        skimage.io.imsave(save_path, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", save_path)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect toothbrush head.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        default='/home/ivpl-d28/Pycharmprojects/Mask-R-CNN/detect_code/dataset/train_img/',
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        default='/home/ivpl-d28/Pycharmprojects/Mask-R-CNN/mask_rcnn_coco.h5',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        default='./dataset/test_img',
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()




    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LeisonConfig()
    else:
        class InferenceConfig(LeisonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        # detect_and_color_splash(model, image_path=args.image,
        #                         video_path=args.video)

        # each image in folder
        image_path = args.image
        dirs = os.listdir(image_path)
        print(dirs)
        images = [file for file in dirs if file.endswith('.jpg')]
        for img in images:
            imgname = os.path.join(image_path, img)
            onlyname, _ = os.path.splitext(img)
            imgname_png = onlyname + '.jpg'
            # output_imgname = os.path.join(image_path, imgname_png)
            detect_and_color_splash(model, image_path=imgname, video_path=args.video, img_file_name=imgname_png)

        submission = pd.DataFrame(results)
        print(submission.shape)
        print(submission.head())

        submission.to_csv('./1201_mrcnn_epoch30.csv', index=False)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
