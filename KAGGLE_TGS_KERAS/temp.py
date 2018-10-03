#-*-coding:utf-8-*-
# license by liberyu
# from model_file import get_unet_resnet
from keras.preprocessing import image
#image.ImageDataGenerator
#from keras.utils.vis_utils import plot_model
#model = get_unet_resnet(input_shape=(256,256,3))
#model.summary()
# plot_model(model, 'resnet_unet.png', show_shapes=True)
import os
import pandas as pd
import gc
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import image
import keras.backend as K
# from model_file import get_unet_resnet
# from new_data_load import load_data
# from iou_metric import my_iou_metric_2,my_iou_metric
# #ROOT_DIR = r'C:\Programming\kaggle_salt_challenge'
SEED = 42
import os
import sys
import random
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

# %matplotlib inline

import cv2
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook #, tnrange
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, add, SeparableConv2D
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
import time
t_start = time.time()


###  2  hv_flip \ no brightness  \ no rotation \ no crop and padding 101 \ my_metirc_iou_2
#  \ thresholds > 0.5  \ TTA hflip vflip \ no elu + 1 \more epoch
###  resize 202 no thresholds

version = 12
tmp = 0

save_basic_name = f'Unet_resnet_v{version}.{tmp}'
save_model_name = save_basic_name + '.model'
submission_file = save_basic_name + '.csv'

print('-*-'*30)
print('SAVE MODEL:'+save_model_name)
print('SUBMISSION FILE:'+submission_file)
print('-*-'*30)


img_size_ori = 101
img_size_target = 202


def upsample(img):  # not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):  # not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

# Loading of training/testing ids and depths
train_df = pd.read_csv("./data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

print(len(train_df))

train_df["images"] = [np.array(load_img("./data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("./data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(train_df.coverage, kde=False, ax=axs[0])
sns.distplot(train_df.coverage_class, bins=10, kde=False, ax=axs[1])
plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class")
# plt.show()
#Plotting the depth distributions¶

sns.distplot(train_df.z, label="Train")
sns.distplot(test_df.z, label="Test")
plt.legend()
plt.title("Depth distribution")
# plt.show()
# Create train/validation split stratified by salt coverage

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.tolist()).reshape(-1, 101, 101, 1),
    np.array(train_df.masks.tolist()).reshape(-1, 101, 101, 1),
    # np.array(train_df.images.map(transfer).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    # np.array(train_df.masks.map(transfer).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)

def create_generator(train_images, train_masks, val_images, val_masks, batch_size):

    # Creating the training Image and Mask generator
    image_datagen = image.ImageDataGenerator(shear_range=0.2,
                                             rotation_range=10,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             fill_mode='reflect')

    mask_datagen = image.ImageDataGenerator(shear_range=0.2,
                                            rotation_range=10,
                                            zoom_range=0.2,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            fill_mode='reflect')

    # Keep the same seed for image and mask generators so they fit together

    print(train_images.shape)

    image_datagen.fit(train_images, augment=True, seed=SEED)
    mask_datagen.fit(train_masks, augment=True, seed=SEED)

    x = image_datagen.flow(train_images, batch_size=batch_size, shuffle=True, seed=SEED)
    y = mask_datagen.flow(train_masks, batch_size=batch_size, shuffle=True, seed=SEED)

    # Creating the validation Image and Mask generator
    image_datagen_val = image.ImageDataGenerator()
    mask_datagen_val = image.ImageDataGenerator()

    image_datagen_val.fit(val_images, augment=True, seed=SEED)
    mask_datagen_val.fit(val_masks, augment=True, seed=SEED)

    x_val = image_datagen_val.flow(val_images, batch_size=batch_size, shuffle=True, seed=SEED)
    y_val = mask_datagen_val.flow(val_masks, batch_size=batch_size, shuffle=True, seed=SEED)

    train_generator = zip(x, y)
    val_generator = zip(x_val, y_val)

    return train_generator, val_generator


def train_model(model, loss,
                train_generator, val_generator,
                model_name, out_model_path,
                metrics=None,
                epochs=200, patience=10, optim_type='Adam', learning_rate=0.001):

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)

    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
        ModelCheckpoint(os.path.join(out_model_path, '{}.h5'.format(model_name)),
                        monitor='val_my_iou_metric',
                        mode = "max",
                        save_best_only=True,
                        verbose=1),
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        steps_per_epoch=200,
        validation_data=val_generator,
        validation_steps=200,
        verbose=2,
        # workers=4,
        # use_multiprocessing=True,
        callbacks=callbacks)

    #model_h5_name = "{}.h5".format(model_name)
    #model_json_name = save_model(model=model, model_name=model_name, model_save_path=out_model_path)
    # model.save_weights(out_model_path)
    #pd.DataFrame(history.history).to_csv(os.path.join(out_model_path, r'logs.csv'), index=False)
    #del model
    #K.clear_session()
    #gc.collect()
    #print('Training is finished...')

    #return model_json_name, model_h5_name


def save_model(model, model_name, model_save_path):
    """
    Save trained model and weights
    :param model: Fitted model
    :param model_name: Name of the model to save
    :return: Nothing to return
    """

    model_json_name = "{}.json".format(model_name)
    # model_h5_name = "{}.h5".format(model_name)

    model_json = model.to_json()
    json_file = open(os.path.join(model_save_path, model_json_name), "w")
    json_file.write(model_json)
    json_file.close()
    # model.save_weights(os.path.join(model_save_path, model_h5_name))

    return model_json_name #, model_h5_name
if __name__ == '__main__':
    # model = get_unet_resnet((256,256,3))
    # train_images, train_masks, val_images, val_masks = load_data()
    batch_size = 16
    train_generator, val_generator = create_generator(x_train, y_train, x_valid, y_valid, batch_size)
    # model_name = "res_unet"
    # out_model_path = "./checkpoint/"
    # train_model(model, "binary_crossentropy",
    #             train_generator, val_generator,
    #             model_name, out_model_path,
    #             metrics=[my_iou_metric],
    #             epochs=200, patience=10, optim_type='Adam', learning_rate=0.001)
    pass