from __future__ import division
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os, cv2, sys
import numpy as np
from config import *
from wavelet import wav_trans,preprocess_images, preprocess_maps, postprocess_predictions
from model import ml_net_model, loss

if __name__ == '__main__':
    phase = sys.argv[1]
    path1='/home/vriplab9/Desktop/priya/project/images'
    path2='/home/vriplab9/Desktop/priya/project/images_label'
    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10)
    sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model")
    model.compile(sgd, loss,metrics=['accuracy'])

    if phase == 'train':
        print("Training ML-Net")
        model.fit(preprocess_images(path1,480,480),preprocess_maps(path2,480,480), batch_size=b_s, nb_epoch = nb_epoch, show_accuracy = True,verbose=1,validation_split=0.2)
       

    elif phase == "test":
        # path of output folder
        output_folder = ''

        if len(sys.argv) < 2:
            raise SyntaxError
        imgs_test_path = sys.argv[2]

        file_names = [f for f in os.listdir(imgs_test_path) if f.endswith('.jpg')]
        file_names.sort()
        nb_imgs_test = len(file_names)

        print("Load weights ML-Net")
        model.load_weights('mlnet_salicon_weights.pkl')

        print("Predict saliency maps for " + imgs_test_path)
        predictions = model.predict_generator(generator_test(b_s=1, imgs_test_path=imgs_test_path), nb_imgs_test)

        for pred, name in zip(predictions, file_names):
            original_image = cv2.imread(imgs_test_path + name, 0)
            res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
            cv2.imwrite(output_folder + '%s' % name, res.astype(int))

    else:
        raise NotImplementedError
train
