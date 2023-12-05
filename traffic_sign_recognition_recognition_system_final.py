# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

from PIL import Image
import cv2

!pip install tensorflow
!pip install tensorflow keras
!pip install tensorflow sklearn
!pip install tensorflow matplotlib
!pip install tensorflow pandas
!pip install tensorflow pillow
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score
np.random.seed(42)

from matplotlib import style
style.use('fivethirtyeight')

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

model = load_model('/content/drive/MyDrive/GdriveXbot/model1.h5')

image = "user_input/stop.jfif"

import numpy as np

img_path ='/content/drive/MyDrive/GdriveXbot/user_input/rightTurn.png'
img = Image.open(img_path)

img_array = np.array(img)

"""hello"""

# image1 = cv2.imread('/content/gdrive/MyDrive/GdriveXbot/user_input/stop.jfif')
# print(image1)
image_fromarray = Image.fromarray(img_array, 'RGB')
print(image_fromarray)

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

IMG_HEIGHT = 30
IMG_WIDTH = 30
data =[]
resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
data.append(np.array(resize_image))
X_test = np.array(data)
X_test = X_test/255

#pred = model.predict_classes(X_test)
predict_x=model.predict(X_test)
pred=np.argmax(predict_x,axis=1)
#Accuracy with the test data
#print('Test Data accuracy: ',accuracy_score(labels, pred)*100)
indx=pred[0]
print(classes[pred[0]])

