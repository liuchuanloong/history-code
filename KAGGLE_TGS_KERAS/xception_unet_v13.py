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
import keras
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

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


version = 13
tmp = 0
cv_total = 5


save_basic_name = f'Unet_resnet_v{version}.{tmp}'
save_model_name = save_basic_name + '.model'
submission_file = save_basic_name + '.csv'

print('-*-'*30)
print('SAVE MODEL:'+save_model_name)
print('SUBMISSION FILE:'+submission_file)
print('-*-'*30)


img_size_ori = 101
img_size_target = 202


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

#### Reference  from Heng's discussion
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#382657
def get_mask_type(mask):
    border = 10
    outer = np.zeros((101-2*border, 101-2*border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType = cv2.BORDER_CONSTANT, value = 1)

    cover = (mask>0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask*outer) > 0.5).sum():
        return 1 #border
    if np.all(mask==mask[0]):
        return 2 #vertical

    percentage = cover/(101*101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7

def histcoverage(coverage):
    histall = np.zeros((1,8))
    for c in coverage:
        histall[0,c] += 1
    return histall



# Loading of training/testing ids and depths
train_df = pd.read_csv("./data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

print(len(train_df))

train_df["images"] = [np.array(load_img("./data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("./data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

train_df["coverage_class"] = train_df.masks.map(get_mask_type)

train_all = []
evaluate_all = []
skf = StratifiedKFold(n_splits=cv_total, random_state=1234, shuffle=True)
for train_index, evaluate_index in skf.split(train_df.index.values, train_df.coverage_class):
    train_all.append(train_index)
    evaluate_all.append(evaluate_index)
    print(train_index.shape,evaluate_index.shape) # the shape is slightly different in different cv, it's OK

def get_cv_data(cv_index):
    train_index = train_all[cv_index-1]
    evaluate_index = evaluate_all[cv_index-1]
    x_train = np.array(train_df.images[train_index].tolist()).reshape(-1, img_size_ori, img_size_ori, 1)
    y_train = np.array(train_df.masks[train_index].tolist()).reshape(-1, img_size_ori, img_size_ori, 1)
    x_valid = np.array(train_df.images[evaluate_index].tolist()).reshape(-1, img_size_ori, img_size_ori, 1)
    y_valid = np.array(train_df.masks[evaluate_index].tolist()).reshape(-1, img_size_ori, img_size_ori, 1)
    return x_train,y_train,x_valid,y_valid

##################################################################
cv_index = 1
train_index = train_all[cv_index-1]
evaluate_index = evaluate_all[cv_index-1]

print(train_index.shape,evaluate_index.shape)
histall = histcoverage(train_df.coverage_class[train_index].values)
print(f'train cv{cv_index}, number of each mask class = \n \t{histall}')
histall_test = histcoverage(train_df.coverage_class[evaluate_index].values)
print(f'evaluate cv{cv_index}, number of each mask class = \n \t {histall_test}')

fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(24, 6), sharex=True, sharey=True)

# show mask class example
for c in range(8):
    j= 0
    for i in train_index:
        if train_df.coverage_class[i] == c:
            axes[j,c].imshow(np.array(train_df.masks[i])  )
            axes[j,c].set_axis_off()
            axes[j,c].set_title(f'class {c}')
            axes[j, c].show()
            j += 1
            if(j>=2):
                break
#################################################################

################### transform augmentation  ###########################
from transform import do_resize2, do_center_pad_to_factor2, do_flip_transpose2, \
    do_elastic_transform2, do_horizontal_flip2, do_shift_scale_rotate2, do_random_shift_scale_crop_pad2, \
    do_random_pad_to_factor2, do_horizontal_shear2, do_brightness_multiply, do_brightness_shift, do_gamma, \
    do_invert_intensity
from attrdict import AttrDict

def resize_transform(data):
    img, mask = data[0, :, :], data[1, :, :]
    if not img_size_ori == img_size_target:
        img, mask = do_resize2(img, mask, img_size_target, img_size_target)

    return np.stack((img, mask))



CONFIG = AttrDict({'doresize': True, 'do_flip_transpose': True,
                   'do_center_pad_to_factor': False,
                   'do_random_shift_scale_crop_pad': True,
                   'do_random_pad_to_factor': False,
                   'do_horizontal_flip': True,
                   'do_horizontal_shear': True,
                   'do_elastic_transform': False,
                   'do_brightness_multiply': True,
                   'do_brightness_shift': False, 'do_gamma': False,
                   'do_shift_scale_rotate': False,
                   'do_invert_intensity': False
                   })


class DataGenerator(keras.utils.Sequence):

    def __init__(self, images, masks, batch_size = 8,shuffle = False):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return int(np.floor(len(self.images)/self.batch_size))

    def __getitem__(self, index):

        if index < int(np.floor(len(self.images)/self.batch_size)):
            self.batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
            self.batch_masks = self.masks[index * self.batch_size:(index + 1) * self.batch_size]
            self.data = np.stack((self.batch_images, self.batch_masks), axis=1)
            self.transform_data = np.array([self.transform(x) for x in self.data])
            self.transform_batch_images = self.transform_data[:, :, :, 0].reshape(-1, img_size_target, img_size_target,
                                                                                  1)
            self.transform_batch_masks = self.transform_data[:, :, :, 1].reshape(-1, img_size_target, img_size_target,
                                                                                 1)
            return (self.transform_batch_images, self.transform_batch_masks)
        else:
            end_batch_index = len(self.images) % self.batch_size
            self.batch_images = self.images[(index+1)*self.batch_size:(index+1)*self.batch_size+end_batch_index+1]
            self.batch_images = self.masks[(index+1)*self.batch_size:(index+1)*self.batch_size+end_batch_index+1]

            self.data = np.stack((self.batch_images, self.batch_masks), axis=1)
            self.transform_data = np.array([self.transform(x) for x in self.data])
            self.transform_batch_images = self.transform_data[:, :, :, 0].reshape(-1, img_size_target, img_size_target,
                                                                                  1)
            self.transform_batch_masks = self.transform_data[:, :, :, 1].reshape(-1, img_size_target, img_size_target,
                                                                                 1)
            return (self.transform_batch_images, self.transform_batch_masks)

    def transform(self,data):
        config = CONFIG
        type = np.random.choice(2)
        img, mask = data[0,:,:,0], data[1,:,:,0]
        if type==0:
            img, mask = img, mask
        if config.do_random_shift_scale_crop_pad and type==1:
            img, mask = do_random_shift_scale_crop_pad2(img, mask)
        if config.do_horizontal_shear and type==2:
            img, mask = do_horizontal_shear2(img, mask)
        if config.do_brightness_multiply and type==3:
            img = do_brightness_multiply(img, np.random.uniform(0.92, 1.08))
        if config.do_horizontal_flip and type == 4:
            img, mask = do_horizontal_flip2(img, mask)
        if config.do_center_pad_to_factor and type==5:
            img, mask = do_center_pad_to_factor2(img, mask)
        if config.do_random_pad_to_factor and type==6:
            img, mask = do_random_pad_to_factor2(img, mask)
        if config.do_shift_scale_rotate and type==7:
            img, mask = do_shift_scale_rotate2(img, mask)
        if config.do_elastic_transform and type==8:
            img, mask = do_elastic_transform2(img, mask)
        if config.do_flip_transpose and type==9:
            img, mask = do_flip_transpose2(img, mask, type=np.random.choice(5))
        if config.do_brightness_shift and type==10:
            img = do_brightness_shift(img)
        if config.do_gamma and type==11:
            img = do_gamma(img)
        if config.do_invert_intensity and type==12:
            img = do_invert_intensity(img)
        if not img_size_ori == img_size_target and config.doresize:
            img, mask = do_resize2(img, mask, img_size_target, img_size_target)
        return np.stack((img, mask)).transpose(1, 2, 0)

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

###################  end  transform  define  #############################

#################  define model ######################################

def BatchActivate(x):
    x = BatchNormalization()(x)
    # x = Lambda(lambda x: K.elu(x) + 1)(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def build_model(input_layer, start_neurons=64, DropoutRatio=0.5):


    x = Conv2D(32, (3, 3), activation=None, padding="same")(input_layer)
    x = residual_block(x, 32)
    d1 = residual_block(x, 32, True)
    x = MaxPooling2D((2, 2))(d1)
    # 202->101

    x = Conv2D(64, (3, 3), activation=None, padding="same")(x)
    x = residual_block(x, 64, True)
    # x = Conv2D(64, (3, 3), activation=None, padding="valid")(x)
    d2 = residual_block(x, 64, True)
    # 101->101

    residual = Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(d2)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(d2)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    d3 = add([x, residual])
    # 101-> 51

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(d3)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(d3)
    x = SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    d4 = add([x, residual])
    #51->26

    residual = Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(d4)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(d4)
    x = SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    d5 = add([x, residual])
    #26->13
    x = d5
    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = add([x, residual])
    # 13->7

    x = SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)


    # 7 -> 13
    # deconv5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="valid")(x)
    deconv5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(x)
    deconv5 = Conv2D(start_neurons * 8, (2, 2), activation=None, padding="valid")(deconv5)
    deconv5 = BatchNormalization()(deconv5)
    uconv5 = concatenate([deconv5, d5])
    uconv5 = Dropout(DropoutRatio)(uconv5)

    uconv5 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv5)
    uconv5 = residual_block(uconv5, start_neurons * 8)
    uconv5 = residual_block(uconv5, start_neurons * 8, True)

    # 13 -> 26
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
    # deconv4 = Conv2DTranspose(start_neurons * 8, (2, 2), strides=(2, 2), padding="valid")(uconv5)
    uconv4 = concatenate([deconv4, d4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)

    # 26 -> 51
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    deconv3 = Conv2D(start_neurons * 8, (2, 2), activation=None, padding="valid")(deconv3)
    deconv3 = BatchNormalization()(deconv3)
    uconv3 = concatenate([deconv3, d3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # 51 -> 101
    # deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="valid")(uconv3)
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2 = Conv2D(start_neurons * 2, (2, 2), activation=None, padding="valid")(deconv2)
    deconv2 = BatchNormalization()(deconv2)
    uconv2 = concatenate([deconv2, d2])

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # 101 -> 202
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, d1])

    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer

#####################   my_iou_metric   #####################
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        #         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
        #             metric.append(0)
        #             continue
        #         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
        #             metric.append(0)
        #             continue
        #         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
        #             metric.append(1)
        #             continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0], tf.float64)


##################################################################
# code download from: https://github.com/bermanmaxim/LovaszSoftmax
##################################################################
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss

def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_hreflect =  np.array([np.fliplr(x) for x in x_test])
    # x_test_vreflect =  np.array([np.flipud(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_hreflect).reshape(-1, img_size_target, img_size_target)
    # preds_test3_refect = model.predict(x_test_vreflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    # preds_test += np.array([ np.flipud(x) for x in preds_test3_refect] )
    return preds_test/2



##################   class model  #################################
# input_layer = Input((img_size_target, img_size_target, 1))
# output_layer = build_model(input_layer, 16,0.5)
#
# model1 = Model(input_layer, output_layer)
#
# c = optimizers.adam(lr = 0.01)
# model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
#
# model1 = model1.load_weights('xception.h5', by_name=True)
#
#
# model1.summary()
#
# early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=20, verbose=1)
# model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric',
#                                    mode = 'max', save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
#
# epochs = 150
# batch_size = 32
# history = model1.fit(x_train, y_train,
#                     validation_data=[x_valid, y_valid],
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     callbacks=[ model_checkpoint,reduce_lr, TensorBoard(log_dir='mytensorboard/binary')],
#                     verbose=2)




###################    lovasz model    #####################

# training
ious = [0] * cv_total
for cv_index in range(cv_total):
    basic_name = f'Unet_resnet_v{version}.{tmp}_cv{cv_index+1}'
    print('############################################\n', basic_name)
    save_model_name = basic_name + '.model'

    x_train, y_train, x_valid, y_valid = get_cv_data(cv_index + 1)

    # Data augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    train_generator = DataGenerator(x_train, y_train)
    valid_data = np.stack((x_valid, y_valid), axis=1).reshape(-1, 2, 101, 101)
    valid_data = np.array([resize_transform(x) for x in valid_data]).reshape(-1, 2, img_size_target, img_size_target)
    x_valid, y_valid = valid_data[:, 0, :, :].reshape(-1, img_size_target, img_size_target, 1), valid_data[:, 1, :,:].reshape(-1,img_size_target,img_size_target,1)
    print('-*-' * 30)
    # load_version = 11.0
    # load_basic_name = f'Unet_resnet_v{load_version}'
    # load_model_name = load_basic_name + '.model'

    load_model_name = save_model_name
    print('LOAD MODEL:' + load_model_name)
    print('-*-' * 30)

    # model1 = load_model(load_model_name,custom_objects={'my_iou_metric': my_iou_metric})
    # ###remove layter activation layer and use losvasz loss
    # input_x = model1.layers[0].input
    #
    # output_layer = model1.layers[-1].input
    # model = Model(input_x, output_layer)

    model = load_model(load_model_name, custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                        'lovasz_loss': lovasz_loss})
    c = optimizers.adam(lr=0.01)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

    # model.summary()

    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode='max', patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_my_iou_metric_2',
                                       mode='max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode='max', factor=0.5, patience=6, min_lr=0.000001,
                                  verbose=1)
    epochs = 150
    # batch_size = 32

    history = model.fit_generator(generator=train_generator,
                                  epochs=epochs,
                                  steps_per_epoch=800,
                                  validation_data=(x_valid, y_valid),
                                  validation_steps=100,
                                  verbose=2,
                                  # workers=4,
                                  # use_multiprocessing=True,
                                  callbacks=[model_checkpoint, reduce_lr, early_stopping,
                                             TensorBoard(log_dir=f'mytensorboard/lovasz_cv{cv_index}')]
                                  )

    model.load_weights(save_model_name)

    preds_valid = predict_result(model, x_valid, img_size_target)
    ious[cv_index] = get_iou_vector(y_valid, (preds_valid > 0.5))
for cv_index in range(cv_total):
    print(f"cv {cv_index} ious = {ious[cv_index]}")



# ################## predict model ############
# print('-*-'*30)
# # load_version = 11.0
# # load_basic_name = f'Unet_resnet_v{load_version}'
# # load_model_name = load_basic_name + '.model'
#
# load_model_name = save_model_name
# print('LOAD MODEL:'+load_model_name)
# print('-*-'*30)
#
#
# model = load_model(load_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
#                                                    'lovasz_loss': lovasz_loss})
#
#
# preds_valid = predict_result(model,x_valid,img_size_target=img_size_target)
#
#
# # Score the model and do a threshold optimization by the best IoU.
#
# # src: https://www.kaggle.com/aglotero/another-iou-metric
# def iou_metric(y_true_in, y_pred_in, print_table=False):
#     labels = y_true_in
#     y_pred = y_pred_in
#
#     true_objects = 2
#     pred_objects = 2
#
#     #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
#     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))
#     #     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
#     # print(temp1)
#     intersection = temp1[0]
#     # print("temp2 = ",temp1[1])
#     # print(intersection.shape)
#     # print(intersection)
#     # Compute areas (needed for finding the union between all objects)
#     # print(np.histogram(labels, bins = true_objects))
#     area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
#     # print("area_true = ",area_true)
#     area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
#     area_true = np.expand_dims(area_true, -1)
#     area_pred = np.expand_dims(area_pred, 0)
#
#     # Compute union
#     union = area_true + area_pred - intersection
#
#     # Exclude background from the analysis
#     intersection = intersection[1:, 1:]
#     intersection[intersection == 0] = 1e-9
#
#     union = union[1:, 1:]
#     union[union == 0] = 1e-9
#
#     # Compute the intersection over union
#     iou = intersection / union
#
#     # Precision helper function
#     def precision_at(threshold, iou):
#         matches = iou > threshold
#         true_positives = np.sum(matches, axis=1) == 1  # Correct objects
#         false_positives = np.sum(matches, axis=0) == 0  # Missed objects
#         false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
#         tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
#         return tp, fp, fn
#
#     # Loop over IoU thresholds
#     prec = []
#     if print_table:
#         print("Thresh\tTP\tFP\tFN\tPrec.")
#     for t in np.arange(0.5, 1.0, 0.05):
#         tp, fp, fn = precision_at(t, iou)
#         if (tp + fp + fn) > 0:
#             p = tp / (tp + fp + fn)
#         else:
#             p = 0
#         if print_table:
#             print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
#         prec.append(p)
#
#     if print_table:
#         print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
#     return np.mean(prec)
#
#
# def iou_metric_batch(y_true_in, y_pred_in):
#     batch_size = y_true_in.shape[0]
#     metric = []
#     for batch in range(batch_size):
#         value = iou_metric(y_true_in[batch], y_pred_in[batch])
#         metric.append(value)
#     return np.mean(metric)
#
# ## Scoring for last model, choose threshold by validation data
# thresholds_ori = np.linspace(0.4, 0.5, 31)
# # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
# thresholds = np.log(thresholds_ori/(1-thresholds_ori))
#
# # ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
# # print(ious)
# ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
# print(ious)
#
# # instead of using default 0 as threshold, use validation data to find the best threshold.
# threshold_best_index = np.argmax(ious)
# iou_best = ious[threshold_best_index]
# threshold_best = thresholds[threshold_best_index]
# print(threshold_best)
#
# plt.plot(thresholds, ious)
# plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
# plt.xlabel("Threshold")
# plt.ylabel("IoU")
# plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
# plt.legend()
# plt.show()
"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


x_test = np.array([(upsample(np.array(load_img("./data/images/{}.png".format(idx), grayscale = True)))) / 255 for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)

# preds_test = predict_result(model,x_test,img_size_target=img_size_target)
# average the predictions from different folds
t1 = time.time()
preds_test = np.zeros(np.squeeze(x_test).shape)
for cv_index in range(cv_total):
    basic_name = f'Unet_resnet_v{version}.{tmp}_cv{cv_index+1}'
    model.load_weights(basic_name + '.model')
    preds_test += predict_result(model, x_test, img_size_target) / cv_total

t2 = time.time()
print(f"Usedtime = {t2-t1} s")

t1 = time.time()
# pred_dict = {idx: rle_encode(np.round(preds_test[i] > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
# pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > 0.5)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
t2 = time.time()

print(f"Usedtime = {t2-t1} s")

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(submission_file)

t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")
