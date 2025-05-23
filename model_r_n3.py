from __future__ import division
from keras.models import Model
from keras.layers.core import Dropout,Reshape, Activation,Dense,Flatten,Masking
from keras.layers import Input, merge
#from keras.layers.core import Masking(mask_value=0.0)
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.regularizers import l2
import keras.backend as K
import h5py
from googlenet_custom_layers import LRN,Round
from config import *
import math
from keras.layers.normalization import BatchNormalization # batch normalisation
from keras.layers.advanced_activations import PReLU,ThresholdedReLU
from keras.layers import Layer

#def get_weights_vgg16(f, id):
#    g = f['layer_{}'.format(id)]
#    return [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]

#%%
#def RCL_block(l):
#        out_num_filters=64
#        filtersize=3
#        
#		   
#        conv1 = Convolution2D(out_num_filters,filtersize,filtersize, border_mode='same')
#        stack1 = conv1(l)   	
#        stack2 = BatchNormalization()(stack1)
#        stack3 = PReLU()(stack2)
#        
#        conv2 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal')
#        stack4 = conv2(stack3)
#        stack5 = merge([stack1, stack4], mode='sum')
#        stack6 = BatchNormalization()(stack5)
#        stack7 = PReLU()(stack6)
#    	
#        conv3 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
#        stack8 = conv3(stack7)
#        stack9 = merge([stack1, stack8], mode='sum')
#        stack10 = BatchNormalization()(stack9)
#        stack11 = PReLU()(stack10)    
#        
#        conv4 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
#        stack12 = conv4(stack11)
#        stack13 = merge([stack1, stack12], mode='sum')
#        stack14 = BatchNormalization()(stack13)
#        stack15 = PReLU()(stack14)    
#        
#        
#            
#        return stack15   
#%%
#class Round(Layer):
#
#    def __init__(self, **kwargs):
#        super(Round, self).__init__(**kwargs)
#
#    def get_output(self, train=False):
#        X = self.get_input(train)
#        return K.round(X)
#
#    def get_config(self):
#        config = {"name": self.__class__.__name__}
#        base_config = super(Round, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))
#%%

def RCL_block(l):
        out_num_filters=96
        filtersize=3
        
		   
        conv1 = Convolution2D(out_num_filters,filtersize,filtersize, border_mode='same')
        stack1 = conv1(l)   	
        stack2 = Activation('relu')(stack1)
        stack3 = LRN()(stack2)
        
        conv2 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal')
        stack4 = conv2(stack3)
        stack5 = merge([stack1, stack4], mode='sum')
        stack6 =Activation('relu')(stack5)
        stack7 =LRN()(stack6)
    	
        conv3 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack8 = conv3(stack7)
        stack9 = merge([stack1, stack8], mode='sum')
        stack10=Activation('relu')(stack9)
        stack11 =LRN()(stack10)    
        
        conv4 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack12 = conv4(stack11)
        stack13 = merge([stack1, stack12], mode='sum')
        stack14 =Activation('relu')(stack13)
        stack15 =LRN()(stack14)    
        
        
            
        return stack15 

#%%

#def get_weights_vgg16(f, id):
#    g = f['layer_{}'.format(id)]
#    return [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]

def ml_net_model(img_rows=shape_r, img_cols=shape_r, downsampling_factor_net=8, downsampling_factor_product=10):
   

#    f = h5py.File("vgg16_weights.h5")
#
#    input_ml_net = Input(shape=(3,img_rows, img_cols))
#
#    #########################################################
#    # FEATURE EXTRACTION NETWORK							#
#    #########################################################
#    weights = get_weights_vgg16(f, 1)
#    conv1_1 = Convolution2D(64, 3, 3, weights=weights, activation='relu', border_mode='same')(input_ml_net)
#    weights = get_weights_vgg16(f, 3)
#    conv1_2 = Convolution2D(64, 3, 3, weights=weights, activation='relu', border_mode='same')(conv1_1)
#    conv1_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv1_2)
#
#    weights = get_weights_vgg16(f, 6)
#    conv2_1 = Convolution2D(128, 3, 3, weights=weights, activation='relu', border_mode='same')(conv1_pool)
#    weights = get_weights_vgg16(f, 8)
#    conv2_2 = Convolution2D(128, 3, 3, weights=weights, activation='relu', border_mode='same')(conv2_1)
#    conv2_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv2_2)
#
#    weights = get_weights_vgg16(f, 11)
#    conv3_1 = Convolution2D(256, 3, 3, weights=weights, activation='relu', border_mode='same')(conv2_pool)
#    weights = get_weights_vgg16(f, 13)
#    conv3_2 = Convolution2D(256, 3, 3, weights=weights, activation='relu', border_mode='same')(conv3_1)
#    weights = get_weights_vgg16(f, 15)
#    conv3_3 = Convolution2D(256, 3, 3, weights=weights, activation='relu', border_mode='same')(conv3_2)
#    conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv3_3)
#
#    weights = get_weights_vgg16(f, 18)
#    conv4_1 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv3_pool)
#    weights = get_weights_vgg16(f, 20)
#    conv4_2 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv4_1)
#    weights = get_weights_vgg16(f, 22)
#    conv4_3 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv4_2)
#    conv4_pool = MaxPooling2D((2, 2), strides=(1, 1), border_mode='same')(conv4_3)
#
#    weights = get_weights_vgg16(f, 25)
#    conv5_1 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv4_pool)
#    weights = get_weights_vgg16(f, 27)
#    conv5_2 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv5_1)
#    weights = get_weights_vgg16(f, 29)
#    conv5_3 = Convolution2D(512, 3, 3, weights=weights, activation='relu', border_mode='same')(conv5_2)
#     f = h5py.File("vgg16_weights.h5")

    input_ml_net = Input(shape=(3,img_rows, img_cols))

    #########################################################
    # FEATURE EXTRACTION NETWORK							#
    #########################################################
    
    conv1_1 = Convolution2D(64, 3, 3, init='glorot_normal',activation='relu', border_mode='same')(input_ml_net)
   
    conv1_2 = Convolution2D(64, 3, 3,init='glorot_normal', activation='relu', border_mode='same')(conv1_1)
    conv1_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv1_2)

    
    conv2_1 = Convolution2D(128, 3, 3,init='glorot_normal', activation='relu', border_mode='same')(RCL_block(conv1_pool))
   
    conv2_2 = Convolution2D(128, 3, 3, init='glorot_normal', activation='relu', border_mode='same')(conv2_1)
    conv2_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv2_2)

   
    conv3_1 = Convolution2D(256, 3, 3,init='glorot_normal', activation='relu', border_mode='same')(conv2_pool)
    
    conv3_2 = Convolution2D(256, 3, 3, init='glorot_normal',activation='relu', border_mode='same')(RCL_block(conv3_1))
   
    conv3_3 = Convolution2D(256, 3, 3,init='glorot_normal',  activation='relu', border_mode='same')(conv3_2)
    conv3_pool = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv3_3)

   
    conv4_1 = Convolution2D(512, 3, 3,init='glorot_normal',  activation='relu', border_mode='same')(conv3_pool)
   
    conv4_2 = Convolution2D(512, 3, 3,init='glorot_normal', activation='relu', border_mode='same')(conv4_1)
   
    conv4_3 = Convolution2D(512, 3, 3,init='glorot_normal',  activation='relu', border_mode='same')(conv4_2)
#    conv4_pool = MaxPooling2D((2, 2), strides=(1, 1), border_mode='same')(conv4_3)

    
#    conv5_1 = Convolution2D(512, 3, 3,init='glorot_normal',  activation='relu', border_mode='same')(conv4_pool)
#    
#    conv5_2 = Convolution2D(512, 3, 3,init='glorot_normal', activation='relu', border_mode='same')(conv5_1)
#   
#    conv5_3 = Convolution2D(512, 3, 3, init='glorot_normal', activation='relu', border_mode='same')(conv5_2)

    #########################################################
    # ENCODING NETWORK										#
    #########################################################
#    concatenated = merge([conv3_pool, conv4_pool, conv5_3], mode='concat', concat_axis=1)
#    dropout = Dropout(0.5)(concatenated)
    dropout=Dropout(0.5)( conv4_3)

#    int_conv = Convolution2D(64, 3, 3, init='glorot_normal', activation='relu', border_mode='same')( dropout)
#    fconv=Flatten()(int_conv)
#    dconv=Dense((30*30))(fconv)
#    aconv=Activation('sigmoid')(dconv)
#    pre_final_conv=Reshape((1,30,30))(aconv)
    

    pre_final_conv = Convolution2D(1, 1, 1, init='glorot_normal', activation='sigmoid')(dropout)
    
#    temp=(pre_final_conv)
    temp=UpSampling2D(size=(2,2))(pre_final_conv)
#    
    
#    conv4_3n= Convolution2D(64, 1, 1,init='glorot_normal',  activation='sigmoid', border_mode='same')(conv4_3)
#    r1_m=merge([temp,conv4_3n],mode='concat',concat_axis=1)
#    r1_r=RCL_block(r1_m)
#    o1=Convolution2D(1, 1, 1,init='glorot_normal', activation='sigmoid')(r1_r)
#    r1_u=UpSampling2D(size=(2,2))((o1))
#    r1_u1=Activation('relu')(r1_u)
#    conv3_3n=Convolution2D(64, 1, 1,init='glorot_normal',  activation='sigmoid', border_mode='same')(conv3_3)
#    r2_m=merge([r1_u,(conv3_3n)],mode='concat',concat_axis=1)
#    r2_r=RCL_block(r2_m)
#    o2=Convolution2D(1, 1, 1,init='glorot_normal', activation='sigmoid')(r2_r)
#    r2_u=UpSampling2D(size=(2,2))(Round()(o2))
#    r2_u1=Activation('relu')(r2_u)
    
    conv3_3n=Convolution2D(64, 1, 1,init='glorot_normal',  activation='sigmoid', border_mode='same')(conv3_3)
    r2_m=merge([temp,(conv3_3n)],mode='concat',concat_axis=1)
    r2_r=RCL_block(r2_m)
    o2=Convolution2D(1, 1, 1,init='glorot_normal', activation='sigmoid')(r2_r)
    r2_u=UpSampling2D(size=(2,2))(o2)
    conv2_2n=Convolution2D(64, 1, 1,init='glorot_normal',  activation='sigmoid', border_mode='same')(conv2_2)
    r3_m=merge([r2_u,conv2_2n],mode='concat',concat_axis=1)
    r3_r=RCL_block(r3_m)
    o3=Convolution2D(1, 1, 1,init='glorot_normal', activation='sigmoid')(r3_r)
    r3_u=UpSampling2D(size=(2,2))(o3)
#    r3_u1=Activation('relu')(r3_u)
    conv1_2n=Convolution2D(64, 1, 1,init='glorot_normal',  activation='sigmoid', border_mode='same')(conv1_2)
    r4_m=merge([r3_u,conv1_2n],mode='concat',concat_axis=1)
    r4_r=RCL_block(r4_m)
    
    

#    output_ml_net=Convolution2D(1, 1, 1, init='glorot_normal')(r4_r)
   
#    output_ml_net_r=Reshape((1,240*240))(output_ml_net)
#    output_ml_net_ra=Activation('softmax')(output_ml_net_r)
#    model = Model(input=[input_ml_net], output=[output_ml_net_ra])
    output_ml_net=Convolution2D(1, 1, 1,init='glorot_normal', activation='sigmoid')(r4_r)
   
    model = Model(input=[input_ml_net], output=[ o3,(output_ml_net)])

    #for layer in model.layers:
    #    print(layer.input_shape, layer.output_shape)
    return model

def loss1(y_true, y_pred):
    max_y = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), shape_r_gt, axis=-1)), shape_c_gt, axis=-1)
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))
def loss2(y_true, y_pred):
    max_y = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),int(math.ceil(shape_r )/4), axis=-1)),int(math.ceil(shape_c )/4), axis=-1)
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))
def loss3(y_true, y_pred):
    max_y = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), int(math.ceil(shape_r)/2), axis=-1)), int(math.ceil(shape_c )/2), axis=-1)
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))
def loss4(y_true, y_pred):
    max_y = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), shape_r, axis=-1)), shape_c, axis=-1)
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))

    


