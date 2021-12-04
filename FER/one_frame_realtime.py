import dlib
import cv2
# from keras.preprocessing.image import img_to_array

import numpy as np
from skimage import transform as trans

import tensorflow as tf

from facial_analysis import FacialImageProcessing

from tensorflow.keras.models import load_model, Model

## landmark 설정
landmark_model = "../models/shape_predictor_68_face_landmarks.dat"
landmark_detector = dlib.shape_predictor(landmark_model)

## 클래스 정의
idx_to_class = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

## 모델 로드
base_model = load_model('../models/affectnet_emotions/mobilenet_7.h5')



# 이미지 전처리
imgProcessing = FacialImageProcessing(False)

INPUT_SIZE = (224, 224)
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


# print(get_iou([10,10,20,20],[15,15,25,25]))
'''
def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = [224, 224]
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    if image_size[1] == 224:
        src[:, 0] += 8.0
    src *= 2
    if landmark is not None:
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        # dst=dst[:3]
        # src=src[:3]
        # print(dst.shape,src.shape,dst,src)
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        # print(M)

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox

        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin // 2, 0)
        bb[1] = np.maximum(det[1] - margin // 2, 0)
        bb[2] = np.minimum(det[2] + margin // 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin // 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]

        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped




def mobilenet_preprocess_input(x, **kwargs):
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x
'''

## create dataset ==> concat function scores

USE_ALL_FEATURES = True

## dlib
detector = dlib.get_frontal_face_detector()


## main
# webcam open
try:
    cap = cv2.VideoCapture(0)

except Exception as e:
        print(str(e))

if cap.isOpened:
    print('camera is opened width: {0}, height: {1}'.format(cap.get(3), cap.get(4)))


while(cap.isOpened()):
    ret, image = cap.read()
    # 카메라가 작동한다면..
    if ret:
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 영상 이미지를 rgb 형식으로 변환

        bounding_boxes, points = imgProcessing.detect_faces(frame)  # 프레임에서 찾은 얼굴 전처리해서 바운딩 박스와 포인트 값으로 반환
        # 행렬을 전치 시키기
        points = points.T

        # loop as the number of face, one loop per one face
        for bbox, p in zip(bounding_boxes, points):
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            face_img = frame[y1:y2, x1:x2, :]

            # 얼굴이 잡히는데 정해진 사이즈보다 클 경우 예외처리
            try:
                face_img = cv2.resize(face_img, INPUT_SIZE)

                inp = face_img.astype(np.float32)
                inp[..., 0] -= 103.939
                inp[..., 1] -= 116.779
                inp[..., 2] -= 123.68
                inp = np.expand_dims(inp, axis=0)
                scores = base_model.predict(inp)[0] ## error
                print(base_model.predict(inp))
                print(scores)

                emotion = idx_to_class[np.argmax(scores)]

                p = p.reshape((2, 5)).T

                # top, left, bottom, right = box[0:4]
                # face=dlib.rectangle(left, top, right, bottom)
                face = dlib.rectangle(x1, y1, x2, y2)

                #face_img = preprocess(frame, box, p)  ## CROPPED AND ALIGNED
                #face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                ##show aligned cropped image
                #cv2.imshow("face_img", face_img)

                ''' landmark detection '''
                lanmark(image, face)

                ### draw bounding box on original image
                cv2.rectangle(image, (face.left() - 5, face.top() - 5), (face.right() + 5, face.bottom() + 5),
                              (0, 186, 255), 3)

                cv2.putText(image, emotion, (int(face.left()) + 10, int(face.top()) - 10), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(str(e))


        cv2.imshow("original", image)


    else:
        print("error")

    if cv2.waitKey(25) & 0xFF == ord('q'):
        record = False
        break

cap.release()
cv2.destroyAllWindows()
