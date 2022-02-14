
import os


# define dir
AFEW_TRAIN_DIR = '/FER/AFEW/Train_AFEW/'
AFEW_TRAIN_LBPTOP = '/FER/AFEW/Train_AFEW/AlignedFaces_LBPTOP_Points/AlignedFaces_LBPTOP_Points/'
AFEW_TRAIN_OUT_DIR = '//FER/AFEW/Train_AFEW/AlignedFaces_LBPTOP_Points/AlignedFaces_LBPTOP_Points/frames/'


## Extract frames
for emotion in ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']:
    for filename in os.listdir(AFEW_TRAIN_DIR + emotion):
        if not os.path.exists(AFEW_TRAIN_OUT_DIR + emotion + "/" + str(os.path.splitext(filename)[0])):
            print(f"{filename} not exist!")
            os.mkdir(AFEW_TRAIN_OUT_DIR + emotion + "/" + str(os.path.splitext(filename)[0]))
            
      # -i => input file name  -r => frame rate (Hz default = 25)
      command = "ffmpeg -r 1 -i " + AFEW_TRAIN_DIR + emotion + "/" + str(filename) +  " -r 1 " + AFEW_TRAIN_OUT_DIR + emotion + "/" + str(os.path.splitext(filename)[0]) + "/%03d.png"

      os.system(command=command)
