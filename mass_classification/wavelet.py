from __future__ import division
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten,Reshape
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import theano
import os
from PIL import Image
from numpy import *
import numpy as np
import collections
from sklearn.utils import shuffle
import keras.optimizers
from sklearn.feature_extraction import image
import pywt
import cv2
import numpy as np


#%%
def wav_trans(path1):
    img_rows,img_cols=960,480
    listing1=os.listdir(path1)
    num_samples=size(listing1)
    wav_ar=[]
    for file in listing1:
        im1=Image.open(path1+"//"+file)
        img1=im1.resize((img_rows,img_cols))
        gray1=img1.convert('L')
        a_v,a_d=a,b= pywt.dwt(gray1, 'db1')
        if len(wav_ar)==0:
            wav_ar=a_v
        else:
            wav_ar=np.concatenate((wav_ar,a_v),axis=1)
    wav_ar_n=reshape(wav_ar,(num_samples,480,480))
            
    return wav_ar_n       
#%%
def preprocess_images(paths, shape_r, shape_c):
    listing1=os.listdir(paths)
    num_samples=size(listing1)
    ims = np.zeros((num_samples, shape_r, shape_c, 3))

    
    #listing2=os.listdir(path3)
    count=0
    for file in listing1:
        original_image=array(Image.open(paths+"//"+file))
        padded_image = padding(original_image, shape_r, shape_c,3)
#        padded_image = resize(original_image, (shape_r, shape_c,3))
        ims[count] = padded_image
        count=count+1
         

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
       
    wav=wav_trans(paths)
    temp=np.zeros((ims.shape[0],480,480,4))
    temp[:,:,:,0:3]=ims
    temp[:,:,:,3]=wav   
    temp = temp.transpose((0, 3, 1, 2))
    return temp
#%%
def padding(img, shape_r=480, shape_c=480, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded
#%%
def preprocess_maps(paths, shape_r, shape_c):
    listing1=os.listdir(paths)
    num_samples=size(listing1)
#    ims = np.zeros((len(paths), 1, shape_r, shape_c))
    ims = np.zeros((num_samples, 1, shape_r, shape_c))
    count=0
    for file in listing1:   
        original_map=array(Image.open(paths+"//"+file))
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[count] = padded_map.astype(np.float32)
        ims[count] /= 255.0
#        print count
        count=count+1   
    return ims


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img / np.max(img) * 255
#%%
#pi=preprocess_images(path1,480,480)
#wav=wav_trans(path1)
#path1='/home/priya/Downloads/images2'
#
#temp=np.zeros((pi.shape[0],480,480,4))
#temp[:,:,:,0:3]=pi
#temp[:,:,:,3]=wav    


#def generator(b_s, phase_gen='train'):
#    if phase_gen == 'train':
#        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith('.jpg')]
#        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith('.png')]
#    elif phase_gen == 'val':
#        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith('.jpg')]
#        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith('.png')]
#    else:
#        raise NotImplementedError
#
#    images.sort()
#    maps.sort()
#
#    counter = 0
#    while True:
#        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r_gt, shape_c_gt)
#        counter = (counter + b_s) % len(images)
#
#
#
