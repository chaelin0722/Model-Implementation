'''
https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/src/AFEW_train.ipynb

위의 코드 참고해서 frame 10장씩 계산하게 하는 코드

REAL-TIME video frames
'''


import dlib
import os
from PIL import Image
import cv2
from sklearn import preprocessing
#from keras.preprocessing.image import img_to_array

import numpy as np
from skimage import transform as trans

import tensorflow as tf
## mtcnn
from sklearn.ensemble import RandomForestClassifier

from facial_analysis import FacialImageProcessing

from tensorflow.keras.models import load_model,Model


## for error
#config = tf.ConfigProto()

#config.gpu_options.allow_growth = True

#tf.Session(config=config)

idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
base_model=load_model('../models/affectnet_emotions/mobilenet_7.h5')


#base_model = load_model('../models/pretrained_faces/age_gender_tf2_224_deep-03-0.13-0.97.h5')
#base_model = torch.load("../models/pretrained_faces/state_vggface2_enet0_new.pt")

feature_extractor_model=Model(base_model.input,[base_model.get_layer('global_pooling').output,base_model.get_layer('feats').output,base_model.output])
feature_extractor_model.summary()
_,w,h,_=feature_extractor_model.input.shape


#afew_model = load_model('C:/Users/ccaa9/PycharmProjects/real-time_recognition/face-emotion-recognition/models/affectnet_emotion/enet_b0_8_best_afew.pt')

# PyTorch의 가중치는 GPU용으로 저장이 되어있는 경우가 많기 때문에 꼭 map_location 인자를 넣어주어야 합니다.
#torch_state_dict = torch.load(afew_model, map_location=torch.device("cpu"))
# 기본적으로 torch.Tensor로 로딩이 됩니다. 따라서 detach()와 numpy() 메소드를 불러주는 것이 꼭 필요합니다.
#torch_state_dict = {key: val.detach().numpy() for key, val in torch_state_dict.items()}


landmark_model = 'shape_predictor_68_face_landmarks.dat'

imgProcessing=FacialImageProcessing(False)
print(tf.__version__)

landmark_detector = dlib.shape_predictor(landmark_model)
emotion_to_index = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Neutral':4, 'Sad':5, 'Surprise':6}
INPUT_SIZE = (224,224)
### extract frames
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : array
        order: {'x1', 'y1', 'x2', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : array
        order: {'x1', 'y1', 'x2', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

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


# landmark detection using dlib
def lanmark(image, face):

    # 얼굴에서 68개 점 찾기
    landmarks = landmark_detector(image, face)

    # create list to contain landmarks
    landmark_list = []

    # append (x, y) in landmark_list
    for p in landmarks.parts():
        landmark_list.append([p.x, p.y])
        cv2.circle(image, (p.x, p.y), 2, (255, 255, 255), -1)

def mobilenet_preprocess_input(x,**kwargs):
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


def get_features_scores(image):
    filename2features = {}
    X_global_features, X_feats, X_scores, X_isface = [], [], [], []
    images = image
    images_10 = []
    i = 0
    for imgs in images:
        X_isface.append(True)  # making bbox means has face! so always have faces

        images_10.append(imgs)
        inp = preprocessing_function(np.array(images_10, dtype=np.float32))
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

    print("global", X_global_features)
    X_isface = np.array(X_isface)
        # print(X_global_features.shape,X_feats.shape,X_scores.shape)

    filename2features[i] = (X_global_features, X_feats, X_scores, X_isface)
    i += 1

    return filename2features

## create dataset ==> concat function scores

USE_ALL_FEATURES = True


def create_dataset(filename2features):
    x = []
    y = []
    has_faces = []
    ind = 0
    features = filename2features[0]
    total_features = None

    if USE_ALL_FEATURES and True:
        print('here')
        #for face in [1, 0]:
        cur_features = features[ind]

        #if len(cur_features) == 0:
        #    continue
        weight = len(cur_features) / len(features[ind])
        mean_features = np.mean(cur_features, axis=0)
        std_features = np.std(cur_features, axis=0)
        max_features = np.max(cur_features, axis=0)
        min_features = np.min(cur_features, axis=0)

        # join several features together
        feature = np.concatenate((mean_features, std_features, min_features, max_features), axis=None)
        print("Feature", feature)
        if total_features is None:
            total_features = weight * feature
        else:
            total_features += weight * feature
    has_faces.append(1)
    print("total_Features : ", total_features)


    if total_features is not None:
        print("totla features is not none")
        x.append(total_features)
        #y.append(emotion_to_index[category])


    print("out of for moon")
    x = np.array(x)
    y = np.array(y)
    has_faces = np.array(has_faces)
    #print("x : ", x.shape, "y :", y.shape, "has_face : ", has_faces)
    #return x, y, has_faces
    return x #, y, has_faces


## dlib
detector = dlib.get_frontal_face_detector()


## main

preprocessing_function=mobilenet_preprocess_input
# webcam open
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

print('camera is opened width: {0}, height: {1}'.format(cap.get(3), cap.get(4)))


if cap.isOpened():
    print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))

while(cap.isOpened()):

    ret, image = cap.read()

    if ret:

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bounding_boxes, points = imgProcessing.detect_faces(frame)
        points = points.T

        faceframe_10 = []
        for i in range(10):
            for bbox, p in zip(bounding_boxes, points):
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]
                # face_img=frame[y1:y2,x1:x2,:]

                p = p.reshape((2, 5)).T

                #top, left, bottom, right = box[0:4]
                #face=dlib.rectangle(left, top, right, bottom)
                face=dlib.rectangle(x1, y1, x2, y2)

                face_img = preprocess(frame, box, p)  ## CROPPED AND ALIGNED
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)


                ###show aligned cropped image
                cv2.imshow("face_img", face_img)

                ### draw bounding box on original image
                cv2.rectangle(image, (face.left() - 5, face.top() - 5), (face.right() + 5, face.bottom() + 5),
                              (0, 186, 255), 3)
                
                cv2.imshow("original", image)

                faceframe_10.append(face_img)
                
        n = 0
        ## feature score threw aligned cropped image
        features_scores = get_features_scores(faceframe_10)
        x_score= create_dataset(features_scores)
        n += 1

        ## normalization
        x_score_norm = preprocessing.normalize(x_score, norm='l2')
        
        #np.random.seed(1)
        #clf = RandomForestClassifier(n_estimators=1000, max_depth=7, n_jobs=-1)

        ## ? how to use x_score_norm to predict emotion??



    else:
        print("error")


    if cv2.waitKey(25) & 0xFF == ord('q'):
        record = False
        break


cap.release()
cv2.destroyAllWindows()
