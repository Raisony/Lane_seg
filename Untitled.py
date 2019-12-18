#!/usr/bin/env python
# coding: utf-8

# In[7]:


from __future__ import print_function
import cv2, os, sys
import glob
import numpy as np

# import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
# from google.colab import drive
# drive.mount('/content/drive')
# %cd '/content/drive/My Drive/lanesegment/tf_unet'

import skimage.io as io
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras import optimizers
from keras import metrics
from keras import losses

from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 800, 800, 3
TRAIN_PATH = '../membrane/Trainset/'
TEST_PATH = '../membrane/test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 233
random.seed = seed
np.random.seed = seed


# In[8]:


from models import satellite_unet
from models import vanilla_unet
from models import custom_unet
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)
model1 = satellite_unet(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)) 
model2 = vanilla_unet(input_shape=(800, 800, 1))
model3 = custom_unet(
    input_shape=(800, 800, 1),
    use_batch_norm=False,
    num_classes=2,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid')
Model = [model1, model2, model3]
# model = Sequential()
# model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
# model.add(Activation('softmax'))
# sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=1e-4, beta_1 = 0.95, beta_2 = 0.999, epsilon = None, decay=0.0, amsgrad = False)
for model in Model:
    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=[metrics.binary_accuracy, metrics.sparse_categorical_accuracy])
    model.summary()


# In[ ]:


Trainset_img = sorted(os.listdir('./membrane/Trainingset/Image/'))
Trainset_msk = sorted(os.listdir('./membrane/Trainingset/Label/'))
Testset = os.listdir('./membrane/test/')


de, pos = [], []
for i in Trainset_img:
    if '(1)' in i:
        de.append(i)
for i in Trainset_msk:
    if '(1)' in i:
        pos.append(i)
Trainset_img = [item for item in Trainset_img if item not in de] 
Trainset_msk = [item_ for item_ in Trainset_msk if item_ not in pos] 
len(Trainset_img), len(Trainset_msk)
for i,e in enumerate(Trainset_img):
    if e[3:] != Trainset_msk[i][3:]:
        print(e, Trainset_msk[i])


# In[ ]:


T = len(Trainset_img)
# im_g = imread('/content/drive/My Drive/lanesegment/tf_unet/membrane/Trainset/Label/' + Trainset_msk[15])[:,:,:1]
# plt.show(im_g)
# IMG_HEIGHT, IMG_WIDTH = 3000, 3000
X_train = np.zeros((T, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
Y_train = np.zeros((T, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
# Y_train_2 = np.zeros((421, 612, 612, 1), dtype=np.uint8)
# Y_train_3 = np.zeros((421, 800, 800, 2), dtype=np.uint8)
print('Getting and resizing train images and masks ... ')
for n, id_ in (enumerate(Trainset_img)):
    img = imread('/content/drive/My Drive/lanesegment/tf_unet/membrane/Trainingset/Image/' + id_)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img


# In[ ]:


for n, id_ in enumerate(Trainset_msk):
    mask = imread('/content/drive/My Drive/lanesegment/tf_unet/membrane/Trainingset/Label/' + id_)[:,:,:1]
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask_ = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8) + 75
    mask = np.maximum(mask, mask_)
    L, l = np.where(mask > 75), np.where(mask <= 75)
    mask[L],  mask[l] = True, False
    Y_train[n] = mask


# In[ ]:


X_test = np.zeros((101, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
for n, id_ in enumerate(Testset):
    img = imread('/content/drive/My Drive/lanesegment/tf_unet/membrane/test/' + id_)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img


# In[ ]:


imshow(X_train[486])
plt.show()
imshow(Y_train[486])
plt.show()


# In[ ]:


# earlystopper = EarlyStopping(patience=5, verbose=1)
model = model1
# load_model('model_color_150.h5')
checkpointer = ModelCheckpoint('model_color_1.h5', verbose=1, save_best_only=True)
result = model.fit(X_train, Y_train, validation_split=0.1, batch_size=2, epochs=500, callbacks = [checkpointer])


# In[ ]:





# In[ ]:





# In[ ]:




