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
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
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


###  2  hv_flip \ no brightness  \ no rotation \ no crop and padding 101 \ my_metirc_iou_2   \ thresholds > 0.5  \ TTA hflip vflip \ no elu + 1 \more epoch
###  resize 202 no thresholds

version = 10
tmp = 0
load_version = 9

save_basic_name = f'Unet_resnet_v{version}.{tmp}'
load_basic_name = f'Unet_resnet_v{load_version}'
load_model_name = load_basic_name + '.model'
save_model_name = save_basic_name + '.model'
submission_file = save_basic_name + '.csv'

print('-*-'*30)
print('LOAD MODEL:'+load_model_name)
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

################### transform augmentation  ###########################
from transform import do_resize2, do_center_pad_to_factor2, do_flip_transpose2, \
    do_elastic_transform2, do_horizontal_flip2, do_shift_scale_rotate2, do_random_shift_scale_crop_pad2, \
    do_random_pad_to_factor2, do_horizontal_shear2, do_brightness_multiply, do_brightness_shift, do_gamma, \
    do_invert_intensity
from attrdict import AttrDict

def transform(imgs, masks, config):
    transform_imgs = []
    transform_masks = []
    for id in range(len(imgs)):
        img, mask = np.squeeze(imgs[id], axis=2), np.squeeze(masks[id],axis=2)
        if config.do_center_pad_to_factor and np.random.randint(2):
            img, mask = do_center_pad_to_factor2(img, mask)
        if config.do_random_shift_scale_crop_pad:
            img, mask = do_random_shift_scale_crop_pad2(img, mask)
        if config.do_random_pad_to_factor and np.random.randint(2):
            img, mask = do_random_pad_to_factor2(img, mask)
        if config.do_shift_scale_rotate and np.random.randint(2):
            img, mask = do_shift_scale_rotate2(img, mask)
        if config.do_horizontal_flip and np.random.randint(2):
            img, mask = do_horizontal_flip2(img, mask)
        if config.do_horizontal_shear and np.random.randint(2):
            img, mask = do_horizontal_shear2(img, mask)
        if config.do_elastic_transform and np.random.randint(2):
            img, mask = do_elastic_transform2(img, mask)
        if config.do_flip_transpose and np.random.randint(2):
            img, mask = do_flip_transpose2(img, mask, type=np.random.choice(7))
        if config.do_brightness_shift and np.random.randint(2):
            img = do_brightness_shift(img)
        if config.do_brightness_multiply and np.random.randint(2):
            img = do_brightness_multiply(img)
        if config.do_gamma and np.random.randint(2):
            img = do_gamma(img)
        if config.do_invert_intensity and np.random.randint(2):
            img = do_invert_intensity(img)
        if not img_size_ori == img_size_target and config.doresize:
            img, mask = do_resize2(img, mask, img_size_target, img_size_target)

        transform_imgs.append(img)
        transform_masks.append(mask)
    return np.array(transform_imgs).reshape(-1, img_size_target, img_size_target, 1), \
           np.array(transform_masks).reshape(-1, img_size_target, img_size_target, 1)

def valid_transform(imgs, masks, config):
    valid_transform_imgs = []
    valid_trasnform_masks = []
    for id in range(len(imgs)):
        img, mask = np.squeeze(imgs[id], axis=2), np.squeeze(masks[id],axis=2)

        if not img_size_ori == img_size_target and config.doresize:
            img, mask = do_resize2(img, mask, img_size_target, img_size_target)
        valid_transform_imgs.append(img)
        valid_trasnform_masks.append(mask)
    return np.array(valid_transform_imgs).reshape(-1, img_size_target, img_size_target, 1), \
           np.array(valid_trasnform_masks).reshape(-1, img_size_target, img_size_target, 1)


CONFIG = AttrDict({'doresize': True, 'do_flip_transpose': True,
                   'do_center_pad_to_factor': False,
                   'do_random_shift_scale_crop_pad': True,
                   'do_random_pad_to_factor': False,
                   'do_horizontal_flip': False,
                   'do_horizontal_shear': False,
                   'do_elastic_transform': False,
                   'do_brightness_multiply': True,
                   'do_brightness_shift': False, 'do_gamma': False,
                   'do_shift_scale_rotate': False,
                   'do_invert_intensity': False
                   })

###################  end  transform  define  #############################

###################  Data augmentation  ############################
x1_train = x_train
y1_train = y_train
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
# x_train = np.append(x_train, [np.flipud(x) for x in x_train], axis=0)
# y_train = np.append(y_train, [np.flipud(x) for x in y_train], axis=0)
x1_train, y1_train = transform(x1_train,y1_train, CONFIG)
x_valid, y_valid = valid_transform(x_valid, y_valid, CONFIG)
x_train = np.concatenate((x_train, x1_train), axis=0)
y_train = np.concatenate((y_train, y1_train), axis=0)

print('x_train:{}'.format(x_train.shape))
print('y_train:{}'.format(y_train.shape))
print('x_valid:{}'.format(x_valid.shape))
print('y_valid:{}'.format(y_valid.shape))
print('-*-'*30)


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


# Build model
def build_model(input_layer, start_neurons, DropoutRatio=0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # add
    conv5 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool4)
    conv5 = residual_block(conv5, start_neurons * 8)
    conv5 = residual_block(conv5, start_neurons * 8, True)
    pool5 = MaxPooling2D((2, 2))(conv5)
    pool5 = Dropout(DropoutRatio)(pool5)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool5)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    # add
    # deconv5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="valid")(convm)
    deconv5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Dropout(DropoutRatio)(uconv5)

    uconv5 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv5)
    uconv5 = residual_block(uconv5, start_neurons * 8)
    uconv5 = residual_block(uconv5, start_neurons * 8, True)

    # 6 -> 12
    # deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="valid")(uconv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="valid")(uconv3)
    # deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

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





# ##################   class model  #################################
# input_layer = Input((img_size_target, img_size_target, 1))
# output_layer = build_model(input_layer, 16,0.5)
#
# model1 = Model(input_layer, output_layer)
#
# c = optimizers.adam(lr = 0.01)
# model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
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
#
#
#
#
# ###################    lovasz model    #####################
# print('-*-'*30)
# model1 = load_model(load_model_name,custom_objects={'my_iou_metric': my_iou_metric})
# ###remove layter activation layer and use losvasz loss
# input_x = model1.layers[0].input
#
# output_layer = model1.layers[-1].input
# model = Model(input_x, output_layer)


# model = load_model(load_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
#                                                    'lovasz_loss': lovasz_loss})
# c = optimizers.adam(lr = 0.01)
#
# # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
# # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
# model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
#
# #model.summary()
#
# early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
# model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2',
#                                    mode = 'max', save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=6, min_lr=0.000001, verbose=1)
# epochs = 150
# batch_size = 32
#
# history = model.fit(x_train, y_train,
#                     validation_data=[x_valid, y_valid],
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     callbacks=[ model_checkpoint,reduce_lr,early_stopping, TensorBoard(log_dir='mytensorboard/lovasz')],
#                     verbose=2)
#
# fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
# ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
# ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
# ax_loss.legend()
# ax_score.plot(history.epoch, history.history["my_iou_metric_2"], label="Train score")
# ax_score.plot(history.epoch, history.history["val_my_iou_metric_2"], label="Validation score")
# ax_score.legend()




################## predict model ############
load_model_name = 'Unet_resnet_v10.0.model'
print('predict load model:' + load_model_name)

model = load_model(load_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                   'lovasz_loss': lovasz_loss})

def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_hreflect =  np.array([np.fliplr(x) for x in x_test])
    # x_test_vreflect =  np.array([np.flipud(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_hreflect).reshape(-1, img_size_target, img_size_target)
    # preds_test3_refect = model.predict(x_test_vreflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    # preds_test += np.array([ np.flipud(x) for x in preds_test3_refect] )
    return preds_test/2

preds_valid = predict_result(model,x_valid,img_size_target=img_size_target)


# Score the model and do a threshold optimization by the best IoU.

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))
    #     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    # print(temp1)
    intersection = temp1[0]
    # print("temp2 = ",temp1[1])
    # print(intersection.shape)
    # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    # print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    # print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

## Scoring for last model, choose threshold by validation data
thresholds_ori = np.linspace(0.4, 0.5, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori))

# ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
# print(ious)
ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
print(ious)

# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
print(threshold_best)

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()
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

preds_test = predict_result(model,x_test,img_size_target=img_size_target)
t1 = time.time()


# pred_dict = {idx: rle_encode(np.round(preds_test[i] > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
t2 = time.time()

print(f"Usedtime = {t2-t1} s")

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(submission_file)

t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")
