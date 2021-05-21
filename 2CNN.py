
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import datetime
import numpy as np
from PIL import Image

batch_size = 128  #arbitary//
num_classes = 10
epochs = 50