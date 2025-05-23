from __future__ import division
from keras.optimizers import SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, merge,Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
import os, cv2, sys
import numpy as np
from config import *
from utilities4 import preprocess_images, preprocess_maps, postprocess_predictions
from model_r_n2 import ml_net_model,RCL_block,loss1,loss2,loss3,loss4
from sklearn.metrics import classification_report
from PIL import Image
from googlenet_custom_layers import LRN
from scipy.stats import threshold
from sklearn.metrics import roc_auc_score
import math
import theano

learning_rate = 1e-5

#def step_decay(epoch):
#    initial_lrate = learning_rate
#    drop = 0.5
#    epochs_drop = 50.0
#    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
#    if epoch % 50 == 0:
#        print 'new learning rate is: {0} \n'.format(lrate)
#    return lrate


model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10)
#model.load_weights("weights_seg_rotated_final15.hdf5") 
#model.load_weights("weights_new1.hdf5") 
#lrate = LearningRateScheduler(step_decay)
#callbacks_list_lrate = [lrate]

sgd = SGD(lr= learning_rate, momentum=0.95, nesterov=True)
model.compile(loss=[loss3,loss4],optimizer='sgd',metrics=['accuracy','fmeasure','precision','recall','mean_absolute_error'])
#model.compile(loss=[loss2,loss3,loss4],optimizer='sgd',metrics=['accuracy','fmeasure','precision','recall','mean_absolute_error'])
#model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy','fmeasure','precision','recall','mean_absolute_error'])
model.summary()
model.load_weights("weights_inbreast_2.hdf5")
filepath="weights_inbreast_2.hdf5"
#filepath="weights_new1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=False, mode='auto',period=1)
callbacks_list = [checkpoint]        
#a=np.asarray(preprocess_images(imgs_train_path,shape_r, shape_c))
#a=np.reshape(a,(428,3,120,120))
data=preprocess_images(imgs_train_path,shape_r, shape_c)
a = np.asarray(data[0])
a = np.transpose(a,(0,1,3,2))
labels=np.asarray(data[1])
ground=np.asarray(data[2])
m1=preprocess_maps(maps_train_path1,30,30)
m2=preprocess_maps(maps_train_path1,60,60)
m3=preprocess_maps(maps_train_path1,120,120)
#m4=preprocess_maps(maps_train_path,240,240)





model_hist = model.fit(a,[m2,m3] ,batch_size=b_s, nb_epoch=nb_epoch,shuffle=True,verbose=1,validation_split = 0.2,callbacks= callbacks_list)   
#        score = model.evaluate(preprocess_images(imgs_train_path, shape_r, shape_c),preprocess_maps(maps_train_path, shape_r_gt, shape_c_gt), show_accuracy = True, verbose = 1)
#%%
import matplotlib.pyplot as plt
maps = preprocess_maps(maps_val_path,120,120)
#predict = model.predict(preprocess_images(imgs_val_path, shape_r, shape_c))
predict = model.predict(a)
predicted_im = predict[1]
#%%
check = 0
b= predicted_im[check]
plt.imshow(b[0,:,:],cmap='gray')

plt.imshow(a[0,0,:,:],cmap = 'gray')

plt.figure()
plt.imshow(m2[check,0,:,:],cmap='gray')
plt.figure()
plt.imshow(m3[check,0,:,:],cmap='gray')
plt.figure()

#model.save_weights('INbreast_weight_1.hdf5')
#%%

#convout1_f = theano.function([model.get_input(train=False)], model.layers[1].get_output(train=False))
#convolutions = convout1_f(reshaped[img_to_visualize: img_to_visualize+1])
#
#
#
##The non-magical version of the previous line is this:
##get_ipython().magic(u'matplotlib inline')
#imshow = plt.imshow #alias
#plt.title("Image used: #%d (digit=%d)" % (img_to_visualize, y_train[img_to_visualize]))
#imshow(X_train[img_to_visualize])
#
#
#plt.title("First convolution:")
#imshow(convolutions[0][0])
#
#
#print "The second dimension tells us how many convolutions do we have: %s (%d convolutions)" % (
#    str(convolutions.shape),
#    convolutions.shape[1])
#
#
#for i, convolution in enumerate(convolutions[0]):
#    plt.figure()
#    plt.title("Convolution %d" % (i))
#    plt.imshow(convolution)
#%%
#  
from keras import backend as K       
input_im = a[0:1,:,:,:]      

get_19th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[19].output])
layer_output_19 = get_19th_layer_output([input_im,0])[0]

get_37th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[37].output])
layer_output_37 = get_37th_layer_output([input_im,0])[0]

get_38th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[38].output])
layer_output_38 = get_38th_layer_output([input_im,0])[0]

get_58th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[76].output])
layer_output_76 = get_58th_layer_output([input_im,0])[0]

get_95th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[95].output])
layer_output_95 = get_95th_layer_output([input_im,0])[0]
#print("Extracted features witin %0.3f seconds" % (time() - t_final))

# the input image
#input_image=a[0:1,:,:,:]
#print(input_image.shape)

#%%
plt.imshow(input_im[0,0,:,:],cmap ='gray')

output_image_19 = layer_output_19[0,:,:,:]
print(output_image_19.shape)

output_image_37 = layer_output_37[0,:,:,:]
print(output_image_37.shape)

output_image_38 = layer_output_38[0,:,:,:]
print(output_image_38.shape)

output_image_76 = layer_output_76[0,:,:,:]
print(output_image_76.shape)

output_image_95 = layer_output_95[0,:,:,:]
print(output_image_76.shape)


output_final_1 = output_image_19[0,:,:]
#output_final_1 = output_final_1 > 0.4
plt.figure()
plt.imshow(output_final_1,cmap ='gray')

output_final_2 = output_image_37[0,:,:]
#output_final_2 = output_final_2 > 0.4
plt.figure()
plt.imshow(output_final_2,cmap ='gray')

output_final_3 = output_image_38[0,:,:]
#output_final_3 = output_final_3 > 0.4
plt.figure()
plt.imshow(output_final_3,cmap ='gray')

output_final_4 = output_image_76[0,:,:]
#output_final_3 = output_final_3 > 0.4
plt.figure()
plt.imshow(output_final_4,cmap ='gray')

output_final_5 = output_image_95[0,:,:]
#output_final_3 = output_final_3 > 0.4
plt.figure()
plt.imshow(output_final_5,cmap ='gray')

#%%


