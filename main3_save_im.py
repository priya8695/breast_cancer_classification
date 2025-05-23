from __future__ import division
from keras.optimizers import SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, merge,Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
import os, cv2, sys
import numpy as np
from config import *
from utilities3 import preprocess_images, preprocess_maps, postprocess_predictions
from model_r_n2 import ml_net_model,RCL_block,loss1,loss2,loss3,loss4
from sklearn.metrics import classification_report
from PIL import Image
from googlenet_custom_layers import LRN
from scipy.stats import threshold
from sklearn.metrics import roc_auc_score

#%%

model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10)
model.load_weights("weights_seg_rotated_final.hdf5") 
sgd = SGD(lr=1e-5, decay=0.05, momentum=0.85, nesterov=True)
model.compile(loss=[loss3,loss4],optimizer='sgd',metrics=['accuracy','fmeasure','precision','recall','mean_absolute_error'])
#model.compile(loss=[loss2,loss3,loss4],optimizer='sgd',metrics=['accuracy','fmeasure','precision','recall','mean_absolute_error'])
#model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy','fmeasure','precision','recall','mean_absolute_error'])
model.summary()
filepath="weights_seg_rotated_test.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=False, mode='auto',period=1)
callbacks_list = [checkpoint]        
a=preprocess_images(imgs_train_path,shape_r, shape_c)
m1=preprocess_maps(maps_train_path,30,30)
m2=preprocess_maps(maps_train_path,60,60)
m3=preprocess_maps(maps_train_path,120,120)
m4=preprocess_maps(maps_train_path,240,240)
 
model.fit(a,[m3,m4] ,batch_size=b_s, nb_epoch=nb_epoch,shuffle=True,verbose=1,validation_split = 0.2,callbacks= callbacks_list )   
#        score = model.evaluate(preprocess_images(imgs_train_path, shape_r, shape_c),preprocess_maps(maps_train_path, shape_r_gt, shape_c_gt), show_accuracy = True, verbose = 1)
#%%
#predict = model.predict(a[0:30,:,:,:])
maps=preprocess_maps(maps_val_path,120,120)
#maps1=cv2.resize(maps,(1024,1024))
images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith('.png')]
images.sort()
a=preprocess_images(imgs_val_path,shape_r, shape_c)
predict = model.predict(a)
a1=predict[1]
#%%
check = 0
maps=preprocess_maps(maps_val_path,120,120)
b=a1[check]
#b_n=cv2.resize(b[0],(1024,1024))
plt.imshow(b[0,:,:],cmap='gray')
plt.figure()
plt.imshow(maps[check,0,:,:],cmap='gray')
plt.figure()
plt.imshow(a[check,1,:,:],cmap='gray')
plt.figure()
#%%
paths=maps_train_path
images = [paths + f for f in os.listdir(paths) if f.endswith('.png')]
images.sort()
listing=os.listdir(paths)
#%% storing predicted image

test=preprocess_images(imgs_train_path,shape_r, shape_c)
#%%
#ground=np.asarray(preprocess_maps(maps_val_path,120,120))
ground=np.asarray(preprocess_images(maps_val_path,120,120))
output_folder='D:\\Pinaki_IIST_summer\\INbreast\\mass_prediction\\original_images\\'
for i in range(ground.shape[1]):
#    a=predict[1]
#    b=a[i]
    ze=str(0)
    if(i<10):
        name=ze+ze+ze+str(i)+'.png'
    else:
        if(i<100):
            name=ze+ze+str(i)+'.png'
        else:
            name=ze+str(i)+'.png'
    b=ground[0,i,:,:]
    b1=np.multiply(b,255)
  
   
    
    cv2.imwrite(output_folder + '%s' % name, b1)
#%%
output_folder='D:\\Pinaki_IIST_summer\\INbreast\\mass_prediction\\inbreast_predictions\\'
for i in range(predict[0].shape[0]):
    ze=str(0)
    if(i<10):
        name=ze+ze+ze+str(i)+'.png'
    else:
        if(i<100):
            name=ze+ze+str(i)+'.png'
        else:
            name=ze+str(i)+'.png'
    a1=predict[1]
    b=a1[i]
    ret,thresh = cv2.threshold(b,0.1,1.0,0)
    b_n=cv2.resize(thresh[0],(1024,1024))
#    b_int=b_n.astype(uint8)
    b_int=thresh[0].astype(uint8)
    b_mul=np.multiply(b_int,255)
    cv2.imwrite(output_folder + '%s' % name, b_mul)
                
    

#%% reading original images
original_image_path = 'D:\\Pinaki_IIST_summer\\INbreast\\mass_prediction\\original_images\\'
images = [original_image_path + f for f in os.listdir(original_image_path) if f.endswith('.png')]
images.sort()
listing=os.listdir(original_image_path)
num_samples=len(listing)
  
  
orig_ar = np.zeros((num_samples, 1124,1124),np.float32)
#    wav=wav_trans(path1)
def padwithtens(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector    
for i, path in enumerate(images):
    original_image = cv2.imread(path)
    orig_ar[i]=np.lib.pad(original_image[:,:,0],50, padwithtens)
#    orig_ar[i]=original_image[:,:,0]
   
    
   
   

    

#%% storing segments of image
def append_boxes(proto_objects_map,min_area, box_all):
        """Adds to the list all bounding boxes found with the saliency map
 
            A saliency map is used to find objects worth tracking in each
            frame. This information is combined with a mean-shift tracker
            to find objects of relevance that move, and to discard everything
            else.
 
            :param proto_objects_map: proto-objects map of the current frame
            :param box_all: append bounding boxes from saliency to this list
            :returns: new list of all collected bounding boxes
        """
        # find all bounding boxes in new saliency map
        min_cnt_area = min_area
        box_sal = []
        M_all=[]
#        cnt_sal, contours, hierarchy = cv2.findContours(proto_objects_map, 1, 2)
        contours, hierarchy = cv2.findContours(proto_objects_map, 1, 2)
#        cnt_sal, contours, hierarchy = cv2.findContours(proto_objects_map,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # discard small contours
            if cv2.contourArea(cnt) < min_cnt_area:
                continue
 
            # otherwise add to list of boxes found from saliency map
            box = cv2.boundingRect(cnt)
            box_all.append(box)
            M = cv2.moments(cnt)
            M_all.append(M)
 
        return [box_all,M_all]
    
    
x=[]
a=predict[1]
b=a[0]
temp = b[0]
ret,thresh = cv2.threshold(temp,0.1,1.0,0)
z = thresh.astype(uint8)

[box,c]=append_boxes(z,10,x)
box_n=np.asarray(box)
del x
M=c[0]
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
#cv2.circle(temp1, (cX, cY), 7, (255, 255, 255), -1)
#cv2.putText(image, "center", (cX - 20, cY - 20),
#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
	# show the image
cv2.imshow("Image", temp)
cv2.waitKey(0)
 
 

#%%
output_folder1='D:\\Pinaki_IIST_summer\\INbreast\\mass_prediction\\inbreast_segmented_images\\'
min_area=10
for i in range(predict[1].shape[0]):
    a1=predict[1]
    
    b=a1[i]
    
    ret,thresh = cv2.threshold(b,0.1,1.0,0)
    b_n=cv2.resize(thresh[0],(1024,1024))
    b_int=b_n.astype(uint8)
    b_int=np.lib.pad(b_int,50, padwithtens)
    x=[]
    x1=[]
   
    [box,c]=append_boxes(b_int,min_area,x)
    maps_re=cv2.resize(maps[i,0,:,:],(1024,1024))
    maps_re=np.lib.pad(maps_re,50, padwithtens)
    [box_maps,c_maps]=append_boxes(maps_re,min_area,x1)
    M=c_maps[0]
    x_map = int(M["m10"] / M["m00"])
    y_map = int(M["m01"] / M["m00"])
    box_n=np.asarray(box)
    
    box_maps_n=np.asarray(box_maps)
    del x
    del x1
    image=orig_ar[i]
    
    file_names = [f for f in os.listdir(temp_folder) if f.endswith('.png')]
    file_names.sort()
    for j in range(box_n.shape[0]):
        bound=box_n[j]
        
        M_n=c[j]
        x = int(M_n["m10"] / M_n["m00"])
        y = int(M_n["m01"] / M_n["m00"])
        new_image=[]
        
        new_image.append(image[(bound[1]-40):(bound[1]+bound[3]+40),(bound[0]-40):(bound[0]+bound[2]+40)])
          
        ze=str(0)
        if(sqrt((x-x_map)**2+(y-y_map)**2)<100):
            
            name=file_names[i][0:(len(file_names[i])-4)]+'_'+str(j)+'_'+'t'+'.png'
           
        else:
        
          
            name=file_names[i][0:(len(file_names[i])-4)]+'_'+str(j)+'.png'
           
                
        new_image_ar=np.asarray(new_image)     
        
        cv2.imwrite(output_folder1 + '%s' % name, new_image_ar[0])
        del new_image                  
                           
                           
#%%
ar1=np.zeros(predict[1].shape[0])
ar2=np.zeros(predict[1].shape[0])

min_area=10
for i in range(predict[1].shape[0]):
    a1=predict[1]
    b=a1[i]
    ret,thresh = cv2.threshold(b,0.1,1.0,0)
    b_n=cv2.resize(thresh[0],(1024,1024))
    b_int=b_n.astype(uint8)
    x=[]
    x1=[]
   
    [box,c]=append_boxes(b_int,min_area,x)
    maps_re=cv2.resize(maps[i,0,:,:],(1024,1024))
    [box_maps,c_maps]=append_boxes(maps_re,min_area,x1)
    M=c_maps[0]
    x_map = int(M["m10"] / M["m00"])
    y_map = int(M["m01"] / M["m00"])
    box_n=np.asarray(box)
    ar2[i]=box_n.shape[0]
    box_maps_n=np.asarray(box_maps)
    
    del x
    del x1
    image=orig_ar[i]
    file_names = [f for f in os.listdir(imgs_val_path) if f.endswith('.png')]
    file_names.sort()
    for j in range(box_n.shape[0]):
        bound=box_n[j]
        M_n=c[j]
        x = int(M_n["m10"] / M_n["m00"])
        y = int(M_n["m01"] / M_n["m00"])
        new_image=[]
        
        new_image.append(image[(bound[1]-50):(bound[1]+bound[3]+50),(bound[0]-50):(bound[0]+bound[2]+50)])
          
        ze=str(0)
        if(sqrt((x-x_map)**2+(y-y_map)**2)<100):
            
#            name=file_names[i][0:(len(file_names[i])-4)]+'_'+str(j)+'_'+'t'+'.png'
            ar1[i]=1
           
        else:
        
          
            name=file_names[i][0:(len(file_names[i])-4)]+'_'+str(j)+'.png'
            
           
                
        new_image_ar=np.asarray(new_image)     
        
#        cv2.imwrite(output_folder1 + '%s' % name, new_image_ar[0])
        del new_image                  
                           
                                                 
                           
                           
                           
                           
                           
#    a1=predict[1]
#    b=a1[i]
#    ret,thresh = cv2.threshold(b,0.2,1.0,0)
#    b_n=cv2.resize(thresh[0],(1024,1024)
##    b_int=b_n.astype(uint8))
#    b_int=thresh[0].astype(uint8)
#    b_mul=np.multiply(b_int,255)
#    cv2.imwrite(output_folder1 + '%s' % name, b_mul)
#%% 
ret,thresh = cv2.threshold(temp,0.2,1.0,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
print M
 
#%%
#
images = [imgs_val_path  + f for f in os.listdir(imgs_val_path ) if f.endswith('.png')]
