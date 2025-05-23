import math

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# batch size
b_s = 30
# number of rows of input images
shape_r =120
# number of cols of input images
shape_c = 120
# number of rows of predicted maps
shape_r_gt = int(math.ceil(shape_r )/8)
# number of cols of predicted maps
shape_c_gt = int(math.ceil(shape_c )/8)
# number of epochs
nb_epoch = 1000

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = 'C:\\Users\\IIST\\Desktop\\Pinaki&Priya\\mass_segment_vgg\\mass_train_rotated_final\\'
# path of training maps
#maps_train_path = 'C:\\Users\\IIST\\Desktop\\Pinaki&Priya\\mass_segment_vgg\\ground_truth_rotated_final\\'
maps_train_path1='D:\\Pinaki_IIST_summer\\INbreast\\mass_prediction\\inbreast_rotated_ground_truth\\'
## number of training images
#nb_imgs_train = 5168
## path of validation images
#imgs_val_path = 'C:\\Users\\IIST\\Desktop\\Pinaki&Priya\\mass_segment_vgg\\mass_train_enhanced\\'
## path of validation maps
maps_val_path = 'D:\\Pinaki_IIST_summer\\INbreast\\mass_prediction\\test_ground_truth\\'
temp_folder='D:\\Pinaki_IIST_summer\\INbreast\\mass_prediction\\images\\'
#output_folder='C:\\Users\\IIST\\Desktop\\Pinaki_FinalYear\\2_Codes\\3.myCode\\mass_saliency\\predicted_maps\\'
output_folder='D:\\Pinaki_IIST_summer\\INbreast\\mass_prediction\\inbreast_predictions\\'
#output_folder1='C:\\Users\\IIST\\Desktop\\Pinaki&Priya\\mass_segment_vgg\\segmented_images_new\\'
#original_image_path = 'C:\\Users\\IIST\\Desktop\\Pinaki_FinalYear\\2_Codes\3.myCode\\mass_saliency\\final-04-18\\mass_train_main_1024\\'
## number of validation images
#nb_imgs_val = 56
