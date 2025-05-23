from __future__ import division
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
from config import *
import h5py
 
#def wav_trans(a):
#    
#   cA, cD = pywt.dwt2(a[:,:,0], 'db1')
#       
#            
#    return [a_v,a_d]  
 
def padding(img, shape_r=shape_r, shape_c=shape_c, channels=3):
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


def preprocess_images(paths, shape_r, shape_c):
    
    mass= []
#    with h5py.File("original_image.mat") as f:
#        for column in f['segmented']:
#            row_data = []
#            for row_number in range(len(column)):            
#                row_data.append(f[column[row_number]][:]) 
#            mass.append(row_data)
#    return mass
    with h5py.File("data.mat") as f:
        for column in f['data']:
            row_data = []
            for row_number in range(len(column)):            
                row_data.append(f[column[row_number]][:]) 
            mass.append(row_data)
    return mass

#    with h5py.File("images.mat") as f:
#        for column in f['segmented']:
#            row_data = []
#            for row_number in range(len(column)):            
#                row_data.append(f[column[row_number]][:]) 
#            mass.append(row_data)
#    
#    input_sz = row_number + 1
#    X_train = np.zeros((input_sz,120,120,3))
#
#    for i in range(0,input_sz):
#        X_train[i,:,:,:] = np.asarray(mass[0][i],dtype = 'float32').T
#        
#    X_train = X_train.transpose((0,3,1,2))
#    
#    return mass
#            
#            
            
#    images = [paths + f for f in os.listdir(paths) if f.endswith('.png')]
#    images.sort()
#    listing=os.listdir(paths)
#    num_samples=len(listing)
#  
#  
#    ims = np.zeros((num_samples, shape_r, shape_c, 4),np.float32)
#    wav=wav_trans(path1)
#    
#    for i, path in enumerate(images):
#        original_image = cv2.imread(path)
#        original_image1=cv2.resize(original_image[:,:,0],(240,240))
##        kernel = np.ones((6,6),np.uint8)
##        opening = cv2.morphologyEx(original_image1, cv2.MORPH_OPEN, kernel)
##        closing= cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
##
###        data1=cv2.equalizeHist(data_n[i])
##        [a,b1]=wav_trans(closing)
#        [a,b1]=wav_trans(original_image1)
#       
##        padded_image = padding(original_image, shape_r, shape_c, 3)
##        padded_image = np.asarray(cv2.resize(original_image1,( shape_r, shape_c)),np.float32)
##        ims[i,:,:,0:3] = original_image
#        b= b1[0]
#        g= b1[1]
#        r= b1[2]
###
##        b= (padded_image[ :, :, 0]-128)
##        g= (padded_image[ :, :, 1]-128)
##        r= (padded_image[ :, :, 2]-128)
####
##        b= (padded_image[ :, :, 0])
##        g= (padded_image[ :, :, 1])
##        r= (padded_image[ :, :, 2])
#        ims[i,:,:,0]= a-np.mean(a)
#        ims[i,:,:,1] = b 
#        ims[i,:,:,2] = g 
#        ims[i,:,:,3] = r 
##        padded_image[:,:,0] = b - np.average(b)
##        padded_image[:,:,1] = g - np.average(g)
##        padded_image[:,:,2] = r - np.average(r)





        
         
   

def preprocess_maps(paths, shape_r, shape_c):
#   mass=[]
#   with h5py.File("ground_truth.mat") as f:
#        for column in f['gt']:
#            row_data = []
#            for row_number in range(len(column)):            
#                row_data.append(f[column[row_number]][:]) 
#            mass.append(row_data)
#    
#  
#    
#   return mass
    images = [paths + f for f in os.listdir(paths) if f.endswith('.png')]
    images.sort()
    

    listing=os.listdir(paths)
    num_samples=len(listing)
    ims = np.zeros((num_samples, 1, shape_r, shape_c), dtype=np.uint8)
    
    for i, path in enumerate(images):
        original_map = cv2.imread(path)
       
#        padded_map = padding(original_map, shape_r, shape_c, 1)
        padded_map = cv2.resize(original_map[:,:,0],( shape_r, shape_c))
        ims[i] = padded_map/255.0
        
       


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