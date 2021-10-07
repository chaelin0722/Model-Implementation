##evaluate 함수 에러 디버깅중 prediction만 됨#


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import efficientnet.tfkeras
from keras.preprocessing.image import ImageDataGenerator
import time
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout
start = time.time()


data_dir = "/home/ivpl-d28/PycharmProjects/NOAH/dataset/test/test_cropped"

data_filenames = os.listdir(data_dir)


categories = []
for filename in data_filenames:
    category = filename.split('_')[0]
    if category == 'error':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': data_filenames,
    'category': categories
})

df['category'] = df['category'].astype('str')



test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe = df,
    directory = data_dir,
    x_col="filename",
    y_col="category",
    target_size=(32,32),
    batch_size=1,
    class_mode=None
)


model = load_model('./saved_model.h5')
model.summary()


print("-- Evaluate --")
scores = model.evaluate(test_generator, steps = len(test_generator) // 32)

preds = model.predict(
    test_generator,
    steps=len(test_generator.filenames)
)

print("-- Evaluate --")
scores = model.evaluate(test_generator, categories,
    steps=len(test_generator.filenames))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))



print("-- Predict --")
output = model.predict_generator(test_generator, steps=1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)


end = time.time()
total_time = end - start

print(f"{total_time:.5f} sec for dealing with {3150} data")
print("cropping per 1 image : ", int(total_time // 3150), "sec")



'''
#### save to csv file for every images
preds = model.predict_generator(
    test_generator,
    steps=len(test_generator.filenames)
)

image_ids = [name.split('/')[-1] for name in test_generator.filenames]
predictions = preds.flatten()
data = {'filename': image_ids, 'category':predictions}
submission = pd.DataFrame(data)
print(submission.head())

submission.to_csv("./test_result.csv", index=False)
'''
