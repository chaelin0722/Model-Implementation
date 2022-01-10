# train --dataset=./1229_crack_dataset/ --weights=coco --logs=./logs/

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
DEFAULT_LOGS_DIR = "./logs/crack"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################
#  Configurations
############################################################


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
    NUM_CLASSES = 1+ 4 #1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10 #100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class ToothBrushCrackDataset(utils.Dataset):


    def load_toothbrush_crack(self, dataset_dir, data_list): #, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("toothbrush_crack", 0, "up_crack")
        self.add_class("toothbrush_crack", 1, "down_crack")
        self.add_class("toothbrush_crack", 2, "line_crack")
        self.add_class("toothbrush_crack", 3, "line_silgum")

        # Train or validation dataset?
        # assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir)

        ## set absolute json route
        annotations = json.load(open("/home/ivpl-d28/Pycharmprojects/NOAH/dataset/1217_crack_dataset/via_region_data.json"))

        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # set count i
        i = 0
        # Add images
        for a in annotations:
### check!!
            for name in data_list:
                if a['filename'] == name:

                    if type(a['regions']) is dict:
                        polygons = [r['shape_attributes'] for r in a['regions'].values()]
                        objects = [s['region_attributes']['label'] for s in a['regions'].values()]
                    else:
                        polygons = [r['shape_attributes'] for r in a['regions']]
                        objects = [s['region_attributes']['label'] for s in a['regions']]

                    # load_mask() needs the image size to convert polygons to masks.
                    # Unfortunately, VIA doesn't include it in JSON, so we must read
                    # the image. This is only managable since the dataset is tiny.
                    name_dict = {'0': 1, '1': 2, '2': 3, '3': 4}
                    num_ids = [name_dict[a] for a in objects]


                    image_path = os.path.join(dataset_dir, a['filename'])
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]
                    print("image name : ", a['filename'], "ids : ", num_ids)

                    self.add_image(
                        "toothbrush_crack",
                        image_id=a['filename'],  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        num_ids = num_ids)




    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "toothbrush_crack":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "toothbrush_crack":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']  ## added
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

            if p['name'] == 'polygon':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # Get indexes of pixels inside the polygon and set them to 1
            # ---------------------------------------------------------------------------------
            elif p['name'] == 'rect':
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

        # print("load_ mask_num_ids", num_ids)
        # return mask.astype(np.bool_), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "toothbrush_crack":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




def train(model):
    ## setting kfold
    FOLD = 0
    N_FOLDS = 5

    image_dataset = os.listdir(args.dataset)

    kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
    splits = kf.split(image_dataset)  # ideally, this should be multilabel stratification

    train_dataset = []
    valid_dataset = []

    def get_fold():
        for i, (train_index, valid_index) in enumerate(splits):
            if i == FOLD:
                print("k-folding...i = ", i)
                for train_num in range(len(train_index)):
                    train_dataset.append(image_dataset[train_index[train_num]])

                for valid_num in range(len(valid_index)):
                    valid_dataset.append(image_dataset[valid_index[valid_num]])

                print("dataset..! kfold : ", train_dataset, "val : ", valid_dataset)
                return train_dataset, valid_dataset

    ## train!!
    train_data, valid_data = get_fold()

    # Training dataset.
    dataset_train = ToothBrushCrackDataset()
    dataset_train.load_toothbrush_crack(args.dataset, train_data) #, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ToothBrushCrackDataset()
    dataset_val.load_toothbrush_crack(args.dataset,valid_data) #, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")


    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
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
        # bounding box visualize
        class_names = ['bg','1','2','3','4']
        bbox = utils.extract_bboxes(r['masks'])
        file_name_bb = "bb_splash_{}".format(img_file_name)
        save_path_bb = os.path.join(args.image, 'result', file_name_bb)

        ## for check
        #print("class_ids", r['class_ids'])

        visualize.display_instances(save_path_bb,image_path, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])
        # skimage.io.imsave(save_path_bb, bb_splash)
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
                        default='./1229_crack_dataset/',
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        default='coco',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        default='./crack_dataset/test_normal',
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
        config = ToothBrushCrackConfig()
    else:
        class InferenceConfig(ToothBrushCrackConfig):
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
        # print(dirs)
        images = [file for file in dirs if file.endswith('.png') or file.endswith('.bmp')]
        for img in images:
            imgname = os.path.join(image_path, img)
            onlyname, _ = os.path.splitext(img)
            imgname_png = onlyname + '.png'
            # output_imgname = os.path.join(image_path, imgname_png)
            detect_and_color_splash(model, image_path=imgname, video_path=args.video, img_file_name=imgname_png)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

