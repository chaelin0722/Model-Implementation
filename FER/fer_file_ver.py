import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import dlib
from skimage import transform as trans

from tqdm import tqdm

import tensorflow as tf
## mtcnn
from facial_analysis import FacialImageProcessing

from tensorflow.keras.applications import mobilenet
from tensorflow.keras.models import load_model,Model
import tensorflow
import pathlib
import tensorflow as tf
print(tf.__version__)

base_model=load_model('../models/affectnet_emotions/mobilenet_7.h5')
feature_extractor_model=Model(base_model.input,[base_model.get_layer('global_pooling').output,base_model.get_layer('feats').output,base_model.output])
feature_extractor_model.summary()
_,w,h,_=feature_extractor_model.input.shape

DATA_DIR='C:/Users/ccaa9/PycharmProjects/dataset/AFEW/'
emotion_to_index = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Neutral':4, 'Sad':5, 'Surprise':6}



imgProcessing=FacialImageProcessing(False)
print(tf.__version__)

INPUT_SIZE = (224,224)
### extract frames
def get_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

#print(get_iou([10,10,20,20],[15,15,25,25]))

def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = [224,224]
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==224:
        src[:,0] += 8.0
    src*=2
    if landmark is not None:
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        #dst=dst[:3]
        #src=src[:3]
        #print(dst.shape,src.shape,dst,src)
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        #print(M)

    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
              det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin//2, 0)
        bb[1] = np.maximum(det[1]-margin//2, 0)
        bb[2] = np.minimum(det[2]+margin//2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin//2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
              ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else: #do align using landmark
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped

#print(get_iou([10,10,20,20],[15,15,25,25]))
fpath = 'C:/Users/ccaa9/PycharmProjects/dataset/AFEW/val_afew/Neutral/000543854/0000001.jpg'
# fpath='/home/HDD6TB/datasets/emotions/EmotiW/AFEW/val/AlignedFaces_LBPTOP_Points_Val/frames/010255520/017.png'
# fpath='/home/HDD6TB/datasets/emotions/EmotiW/AFEW/val/AlignedFaces_LBPTOP_Points_Val/frames/012705800/030.png'
frame_bgr = cv2.imread(fpath)
plt.imshow(frame_bgr)
plt.figure(figsize=(5, 5))
frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(frame)

bounding_boxes, points = imgProcessing.detect_faces(frame)
points = points.T
for bbox, p in zip(bounding_boxes, points):
    box = bbox.astype(int)  ## np.int ==> int
    x1, y1, x2, y2 = box[0:4]
    # face_img=frame[y1:y2,x1:x2,:]

    # face_img=extract_image_chip(frame,p)
    p = p.reshape((2, 5)).T

    plt.figure(figsize=(5, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    face_img = preprocess(frame, box, None)  # p)
    ax1.set_title('Cropped')
    ax1.imshow(face_img)

    face_img = preprocess(frame, box, p)
    ax2.set_title('Aligned')
    ax2.imshow(face_img)

#plt.show()


## dlib
detector = dlib.get_frontal_face_detector()


def mobilenet_preprocess_input(x,**kwargs):
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


preprocessing_function=mobilenet_preprocess_input

def get_features_scores(data_dir):
    filename2features = {}
    for filename in tqdm(os.listdir(data_dir)):
        frames_dir = os.path.join(data_dir, filename)
        X_global_features, X_feats, X_scores, X_isface = [], [], [], []
        imgs = []
        for img_name in os.listdir(frames_dir):
            img = cv2.imread(os.path.join(frames_dir,img_name)) #(os.path.join(frames_dir,img_name))
            X_isface.append('noface' not in img_name)

            if img.size:
                img = cv2.resize(img, (w, h))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                if len(imgs) >= 512:
                    inp = preprocessing_function(np.array(imgs, dtype=np.float32))
                    global_features, feats, scores = feature_extractor_model.predict(inp)
                    print("global")
                    print(global_features.shape,feats.shape,scores.shape)
                    if len(X_feats) == 0:
                        X_feats = feats
                        X_global_features = global_features
                        X_scores = scores
                    else:
                        X_feats = np.concatenate((X_feats, feats), axis=0)
                        X_global_features = np.concatenate((X_global_features, global_features), axis=0)
                        X_scores = np.concatenate((X_scores, scores), axis=0)

                    imgs = []

        if len(imgs) > 0:
            inp = preprocessing_function(np.array(imgs, dtype=np.float32))
            global_features, feats, scores = feature_extractor_model.predict(inp)
            print(global_features.shape,feats.shape,scores.shape)
            if len(X_feats) == 0:
                X_feats = feats
                X_global_features = global_features
                X_scores = scores
            else:
                X_feats = np.concatenate((X_feats, feats), axis=0)
                X_global_features = np.concatenate((X_global_features, global_features), axis=0)
                X_scores = np.concatenate((X_scores, scores), axis=0)

        X_isface = np.array(X_isface)
        # print(X_global_features.shape,X_feats.shape,X_scores.shape)
        filename2features[filename] = (X_global_features, X_feats, X_scores, X_isface)
    return filename2features

## compute function scores
def create_dataset(filename2features, data_dir):
    x = []
    y = []
    has_faces = []
    ind = 0

#    print("features : ", features)
    for category in emotion_to_index:
        for filename in os.listdir(os.path.join(data_dir, category)):
            fn = os.path.splitext(filename)[0]
            #if not fn in filename2features:
            #    continue
            features = filename2features[fn]
            total_features = None

            if USE_ALL_FEATURES and True:
        #        print('here')
                for face in [1, 0]:
                    cur_features = features[ind][features[-1] == face]
                    if len(cur_features) == 0:
                        continue
                    weight = len(cur_features) / len(features[ind])
                    mean_features = np.mean(cur_features, axis=0)
                    std_features = np.std(cur_features, axis=0)
                    max_features = np.max(cur_features, axis=0)
                    min_features = np.min(cur_features, axis=0)

                    # join several features together
                    feature = np.concatenate((mean_features, std_features, min_features, max_features), axis=None)


                    if total_features is None:
                        total_features = weight * feature
                    else:
                        total_features += weight * feature
                has_faces.append(1)
            else:
                print("else!")
                if USE_ALL_FEATURES:
                    cur_features = features[0][features[-1] == 1]
                else:
                    cur_features = features
                if len(cur_features) == 0:
                    has_faces.append(0)
                    total_features = np.zeros_like(feature)
                else:
                    has_faces.append(1)
                    # mean_features=features.mean(axis=0)
                    mean_features = np.mean(cur_features, axis=0)
                    std_features = np.std(cur_features, axis=0)
                    max_features = np.max(cur_features, axis=0)
                    min_features = np.min(cur_features, axis=0)

                    # join several features together
                    feature = np.concatenate((mean_features, std_features, min_features, max_features), axis=None)
                    # feature = np.concatenate((mean_features, std_features, max_features), axis=None)
                    # feature = np.concatenate((mean_features, min_features, max_features), axis=None)
                    # feature = np.concatenate((mean_features, std_features), axis=None)
                    # feature = np.concatenate((max_features, std_features), axis=None)
                    # feature=max_features
                    # feature=mean_features
                    # feature=cur_features[-1]
                    # feature=np.percentile(cur_features, 100,axis=0)

                total_features = feature

            if total_features is not None:
                print("totla features is not none")
                x.append(total_features)
                y.append(emotion_to_index[category])
    print("out of for moon")
    x = np.array(x)
    y = np.array(y)
    has_faces = np.array(has_faces)
    print("x : ", x.shape, "y :", y.shape, "has_face : ", has_faces)
    return x, y, has_faces



USE_ALL_FEATURES = True
#filename2Allfeatures_train=get_features_scores(os.path.join(DATA_DIR,'train/AlignedFaces_LBPTOP_Points/Faces/')) #_cropped
## cropped aligned picture as input!!
filename2Allfeatures=get_features_scores(os.path.join('C:/dummy/'))

model_name='mymobilenet_7_ft_sgd_model_chaelin_ver'
MODEL2EMOTIW_FEATURES=model_name+'_feat_emotiw.pickle'

#print(MODEL2EMOTIW_FEATURES)
#with open(MODEL2EMOTIW_FEATURES, 'wb') as handle:
#    pickle.dump([filename2Allfeatures], handle, protocol=pickle.HIGHEST_PROTOCOL)


#with open(MODEL2EMOTIW_FEATURES, 'rb') as handle:
#    filename2Allfeatures=pickle.load(handle)
#    print(len(filename2Allfeatures))

x_test, y_test, has_faces_test = create_dataset(filename2Allfeatures, os.path.join('C:/dummy2/'))

print(x_test)

print(y_test)

## main

