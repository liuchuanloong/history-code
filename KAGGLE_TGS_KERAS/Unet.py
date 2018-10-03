from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers import Input
from keras.utils import plot_model
from keras.models import Model
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,SeparableConv2D,add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
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
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, add, SeparableConv2D,Concatenate
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
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

# xception = Xception(weights='imagenet', include_top=False, input_shape=(202, 202, 3))
# https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/unet/models.py

def handle_block_names_old(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names_old(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'1')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'2')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'2')(x)
        x = Activation('relu', name=relu_name+'2')(x)

        return x
    return layer


def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), batchnorm=True, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names_old(stage)
        print(input_tensor)
        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name)(input_tensor)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)
        # x = Conv2D(filters, (2, 2), activation=None, padding="valid")(x)
        # x = BatchNormalization()(x)
        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'2')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name+'2')(x)
        x = Activation('relu', name=relu_name+'2')(x)

        return x
    return layer
# default parameters for convolution and batchnorm layers of ResNet models
# parameters are obtained from MXNet converted model

def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params

def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params
def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def basic_identity_block(filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        x = Add()([x, input_tensor])
        return x

    return layer


def basic_conv_block(filters, stage, block, strides=(2, 2)):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        shortcut = x
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = Add()([x, shortcut])
        return x

    return layer


def usual_conv_block(filters, stage, block, strides=(2, 2)):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        shortcut = x
        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Conv2D(filters*4, (1, 1), name=conv_name + '3', **conv_params)(x)

        shortcut = Conv2D(filters*4, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = Add()([x, shortcut])
        return x

    return layer


def usual_identity_block(filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)
        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Conv2D(filters*4, (1, 1), name=conv_name + '3', **conv_params)(x)

        x = Add()([x, input_tensor])
        return x

    return layer
def build_unet(backbone, classes, last_block_filters, skip_layers,
               n_upsample_blocks=5, upsample_rates=(2,2,2,2,2),
               block_type='upsampling', activation='sigmoid',
               **kwargs):

    input = backbone.input
    x = backbone.output
    # x = Conv2D(512, (2, 2), activation=None, padding="valid")(x)
    # x = BatchNormalization()(x)

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_layers = ([get_layer_number(backbone, l) if isinstance(l, str) else l for l in skip_layers])
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        if i < len(skip_layers):
#             print(backbone.layers[skip_layers[i]])
            skip = backbone.layers[skip_layers[i]].output
        else:
            skip = None
        # skip = Conv2D(512, (3, 3), activation=None, padding="valid")(skip)
        # skip = BatchNormalization()(skip)

        up_size = (upsample_rates[i], upsample_rates[i])
        filters = last_block_filters * 2**(n_upsample_blocks-(i+1))

        x = up_block(filters, i, upsample_rate=up_size, skip=skip, **kwargs)(x)

    if classes < 2:
        activation = 'sigmoid'

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model


import keras.backend as K
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import ZeroPadding2D,SpatialDropout2D
from keras.layers import Dense
from keras.models import Model
from keras.engine import get_source_inputs

import keras
from distutils.version import StrictVersion

if StrictVersion(keras.__version__) < StrictVersion('2.2.0'):
    from keras.applications.imagenet_utils import _obtain_input_shape
else:
    from keras_applications.imagenet_utils import _obtain_input_shape
def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv
K.clear_session()
def get_unet_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = UpSampling2D()(conv5)
    up6 = Conv2D(512, (2, 2), activation=None, padding="valid")(up6)
    up6 = concatenate([up6, conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = UpSampling2D()(conv7)
    up8 = Conv2D(256, (2, 2), activation=None, padding="valid")(up8)
    up8 = concatenate([up8, conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = UpSampling2D()(conv8)
    up9 = Conv2D(64, (2, 2), activation=None, padding="valid")(up9)
    up9 = concatenate([up9, conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = UpSampling2D()(conv9)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    output_layer_noActi = Conv2D(1, (1, 1), activation=None, name="prediction")(conv10)
    x = Activation('sigmoid')(output_layer_noActi)

    model = Model(resnet_base.input, x)
    return model
import warnings

# from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'keras.., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'keras.., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1_0')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

#     x = AveragePooling2D((7, 7), name='avg_pool')(x)

#     if include_top:
#         x = Flatten()(x)
#         x = Dense(classes, activation='softmax', name='fc1000')(x)
#     else:
#         if pooling == 'avg':
#             x = GlobalAveragePooling2D()(x)
#         elif pooling == 'max':
#             x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path,by_name=True)
    return model
def relu6(x):
    return K.relu(x, max_value=6)

model = get_unet_resnet(input_shape=(202,202,1))




# def build_resnet(
#         repetitions=(2, 2, 2, 2),
#         include_top=True,
#         input_tensor=None,
#         input_shape=None,
#         classes=1000,
#         block_type='usual'):
#     # Determine proper input shape
#     input_shape = _obtain_input_shape(input_shape,
#                                       default_size=224,
#                                       min_size=101,
#                                       data_format='channels_last',
#                                       require_flatten=include_top)
#
#     if input_tensor is None:
#         img_input = Input(shape=input_shape, name='data')
#     else:
#         if not K.is_keras_tensor(input_tensor):
#             img_input = Input(tensor=input_tensor, shape=input_shape)
#         else:
#             img_input = input_tensor
#
#     # get parameters for model layers
#     no_scale_bn_params = get_bn_params(scale=False)
#     bn_params = get_bn_params()
#     conv_params = get_conv_params()
#     init_filters = 64
#
#     if block_type == 'basic':
#         conv_block = basic_conv_block
#         identity_block = basic_identity_block
#     else:
#         conv_block = usual_conv_block
#         identity_block = usual_identity_block
#
#     # resnet bottom
#     x = BatchNormalization(name='bn_data_0', **no_scale_bn_params)(img_input)
#     x = ZeroPadding2D(padding=(3, 3))(x)
#     x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0_0', **conv_params)(x)
#     x = BatchNormalization(name='bn0', **bn_params)(x)
#     x = Activation('relu', name='relu0')(x)
#     x = ZeroPadding2D(padding=(1, 1))(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)
#
#     # resnet body
#     for stage, rep in enumerate(repetitions):
#         for block in range(rep):
#
#             filters = init_filters * (2 ** stage)
#
#             # first block of first stage without strides because we have maxpooling before
#             if block == 0 and stage == 0:
#                 x = conv_block(filters, stage, block, strides=(1, 1))(x)
#
#             elif block == 0:
#                 x = conv_block(filters, stage, block, strides=(2, 2))(x)
#
#             else:
#                 x = identity_block(filters, stage, block)(x)
#
#     x = BatchNormalization(name='bn1', **bn_params)(x)
#     x = Activation('relu', name='relu1')(x)
#
#     # resnet top
#     if include_top:
#         x = GlobalAveragePooling2D(name='pool1')(x)
#         x = Dense(classes, name='fc1')(x)
#         x = Activation('softmax', name='softmax')(x)
#
#     # Ensure that the model takes into account any potential predecessors of `input_tensor`.
#     if input_tensor is not None:
#         inputs = get_source_inputs(input_tensor)
#     else:
#         inputs = img_input
#
#     # Create model.
#     model = Model(inputs, x)
#
#     return model
#
#
# weights_collection = [
#     # ResNet34
#     {
#         'model': 'resnet34',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': True,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
#         'name': 'resnet34_imagenet_1000.h5',
#         'md5': '2ac8277412f65e5d047f255bcbd10383',
#     },
#
#     {
#         'model': 'resnet34',
#         'dataset': 'imagenet',
#         'classes': 1000,
#         'include_top': False,
#         'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
#         'name': 'resnet34_imagenet_1000_no_top.h5',
#         'md5': '8caaa0ad39d927cb8ba5385bf945d582',
#     },
# ]
#
#
# def ResNet34(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
#     model = build_resnet(input_tensor=input_tensor,
#                          input_shape=input_shape,
#                          repetitions=(3, 4, 6, 3),
#                          classes=classes,
#                          include_top=include_top,
#                          block_type='basic')
#     model.name = 'resnet34'
#
#     if weights:
#         load_model_weights(weights_collection, model, weights, classes, include_top)
#     return model
#
#
# from keras.utils import get_file
#
#
# def find_weights(weights_collection, model_name, dataset, include_top):
#     w = list(filter(lambda x: x['model'] == model_name, weights_collection))
#     w = list(filter(lambda x: x['dataset'] == dataset, w))
#     w = list(filter(lambda x: x['include_top'] == include_top, w))
#     return w
#
#
# def load_model_weights(weights_collection, model, dataset, classes, include_top):
#     weights = find_weights(weights_collection, model.name, dataset, include_top)
#
#     if weights:
#         weights = weights[0]
#
#         if include_top and weights['classes'] != classes:
#             raise ValueError('If using `weights` and `include_top`'
#                              ' as true, `classes` should be {}'.format(weights['classes']))
#
#         weights_path = get_file(weights['name'],
#                                 weights['url'],
#                                 cache_subdir='models',
#                                 md5_hash=weights['md5'])
#
#         model.load_weights(weights_path, by_name=True)
#
#     else:
#         raise ValueError('There is no weights for such configuration: ' +
#                          'model = {}, dataset = {}, '.format(model.name, dataset) +
#                          'classes = {}, include_top = {}.'.format(classes, include_top))
# def UResNet34(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='transpose',
#                        encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):
#
# #     backbone = ResnetBuilder.build_resnet_34(input_shape=input_shape,input_tensor=input_tensor)
#     backbone = ResNet34(input_shape=input_shape, weights='imagenet', classes=1000,include_top=False)
#     skip_connections = list([106,74,37,5])  # for resnet 34
#     model = build_unet(backbone, classes, decoder_filters,
#                        skip_connections, block_type=decoder_block_type,
#                        activation=activation, **kwargs)
#     model.name = 'u-resnet34'
#
#     return model
# model = UResNet34(input_shape=(202,202,1))
# plot_model(model, to_file='model.png')
# # model.summary()




















# def build_model(input_layer, start_neurons=64, DropoutRatio=0.5):
#
#
#     x = Conv2D(32, (3, 3), activation=None, padding="same")(input_layer)
#     x = residual_block(x, 32)
#     d1 = residual_block(x, 32, True)
#     x = MaxPooling2D((2, 2))(d1)
#     # 202->101
#
#     x = Conv2D(64, (3, 3), activation=None, padding="same")(x)
#     x = residual_block(x, 64, True)
#     # x = Conv2D(64, (3, 3), activation=None, padding="valid")(x)
#     d2 = residual_block(x, 64, True)
#     # 101->101
#
#     residual = Conv2D(128, (1, 1),
#                              strides=(2, 2),
#                              padding='same',
#                              use_bias=False)(d2)
#     residual = BatchNormalization()(residual)
#
#     x = SeparableConv2D(128, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block2_sepconv1')(d2)
#     x = BatchNormalization(name='block2_sepconv1_bn')(x)
#     x = Activation('relu', name='block2_sepconv2_act')(x)
#     x = SeparableConv2D(128, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block2_sepconv2')(x)
#     x = BatchNormalization(name='block2_sepconv2_bn')(x)
#
#     x = MaxPooling2D((3, 3),
#                             strides=(2, 2),
#                             padding='same',
#                             name='block2_pool')(x)
#     d3 = add([x, residual])
#     # 101-> 51
#
#     residual = Conv2D(256, (1, 1), strides=(2, 2),
#                              padding='same', use_bias=False)(d3)
#     residual = BatchNormalization()(residual)
#
#     x = Activation('relu', name='block3_sepconv1_act')(d3)
#     x = SeparableConv2D(256, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block3_sepconv1')(x)
#     x = BatchNormalization(name='block3_sepconv1_bn')(x)
#     x = Activation('relu', name='block3_sepconv2_act')(x)
#     x = SeparableConv2D(256, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block3_sepconv2')(x)
#     x = BatchNormalization(name='block3_sepconv2_bn')(x)
#
#     x = MaxPooling2D((3, 3), strides=(2, 2),
#                             padding='same',
#                             name='block3_pool')(x)
#     d4 = add([x, residual])
#     #51->26
#
#     residual = Conv2D(728, (1, 1),
#                              strides=(2, 2),
#                              padding='same',
#                              use_bias=False)(d4)
#     residual = BatchNormalization()(residual)
#
#     x = Activation('relu', name='block4_sepconv1_act')(d4)
#     x = SeparableConv2D(728, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block4_sepconv1')(x)
#     x = BatchNormalization(name='block4_sepconv1_bn')(x)
#     x = Activation('relu', name='block4_sepconv2_act')(x)
#     x = SeparableConv2D(728, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block4_sepconv2')(x)
#     x = BatchNormalization(name='block4_sepconv2_bn')(x)
#
#     x = MaxPooling2D((3, 3), strides=(2, 2),
#                             padding='same',
#                             name='block4_pool')(x)
#     d5 = add([x, residual])
#     #26->13
#     x = d5
#     for i in range(8):
#         residual = x
#         prefix = 'block' + str(i + 5)
#
#         x = Activation('relu', name=prefix + '_sepconv1_act')(x)
#         x = SeparableConv2D(728, (3, 3),
#                                    padding='same',
#                                    use_bias=False,
#                                    name=prefix + '_sepconv1')(x)
#         x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
#         x = Activation('relu', name=prefix + '_sepconv2_act')(x)
#         x = SeparableConv2D(728, (3, 3),
#                                    padding='same',
#                                    use_bias=False,
#                                    name=prefix + '_sepconv2')(x)
#         x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
#         x = Activation('relu', name=prefix + '_sepconv3_act')(x)
#         x = SeparableConv2D(728, (3, 3),
#                                    padding='same',
#                                    use_bias=False,
#                                    name=prefix + '_sepconv3')(x)
#         x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)
#
#         x = add([x, residual])
#
#     residual = Conv2D(1024, (1, 1), strides=(2, 2),
#                              padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)
#
#     x = Activation('relu', name='block13_sepconv1_act')(x)
#     x = SeparableConv2D(728, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block13_sepconv1')(x)
#     x = BatchNormalization(name='block13_sepconv1_bn')(x)
#     x = Activation('relu', name='block13_sepconv2_act')(x)
#     x = SeparableConv2D(1024, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block13_sepconv2')(x)
#     x = BatchNormalization(name='block13_sepconv2_bn')(x)
#
#     x = MaxPooling2D((3, 3),
#                             strides=(2, 2),
#                             padding='same',
#                             name='block13_pool')(x)
#     x = add([x, residual])
#     # 13->7
#
#     x = SeparableConv2D(1536, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block14_sepconv1')(x)
#     x = BatchNormalization(name='block14_sepconv1_bn')(x)
#     x = Activation('relu', name='block14_sepconv1_act')(x)
#
#     x = SeparableConv2D(2048, (3, 3),
#                                padding='same',
#                                use_bias=False,
#                                name='block14_sepconv2')(x)
#     x = BatchNormalization(name='block14_sepconv2_bn')(x)
#     x = Activation('relu', name='block14_sepconv2_act')(x)
#
#
#     # 7 -> 13
#     # deconv5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="valid")(x)
#     deconv5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(x)
#     deconv5 = Conv2D(start_neurons * 8, (2, 2), activation=None, padding="valid")(deconv5)
#     deconv5 = BatchNormalization()(deconv5)
#     uconv5 = concatenate([deconv5, d5])
#     uconv5 = Dropout(DropoutRatio)(uconv5)
#
#     uconv5 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv5)
#     uconv5 = residual_block(uconv5, start_neurons * 8)
#     uconv5 = residual_block(uconv5, start_neurons * 8, True)
#
#     # 13 -> 26
#     deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
#     # deconv4 = Conv2DTranspose(start_neurons * 8, (2, 2), strides=(2, 2), padding="valid")(uconv5)
#     uconv4 = concatenate([deconv4, d4])
#     uconv4 = Dropout(DropoutRatio)(uconv4)
#
#     uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
#     uconv4 = residual_block(uconv4, start_neurons * 8)
#     uconv4 = residual_block(uconv4, start_neurons * 8, True)
#
#     # 26 -> 51
#     deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
#     # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
#     deconv3 = Conv2D(start_neurons * 8, (2, 2), activation=None, padding="valid")(deconv3)
#     deconv3 = BatchNormalization()(deconv3)
#     uconv3 = concatenate([deconv3, d3])
#     uconv3 = Dropout(DropoutRatio)(uconv3)
#
#     uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
#     uconv3 = residual_block(uconv3, start_neurons * 4)
#     uconv3 = residual_block(uconv3, start_neurons * 4, True)
#
#     # 51 -> 101
#     # deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="valid")(uconv3)
#     deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
#     deconv2 = Conv2D(start_neurons * 2, (2, 2), activation=None, padding="valid")(deconv2)
#     deconv2 = BatchNormalization()(deconv2)
#     uconv2 = concatenate([deconv2, d2])
#
#     uconv2 = Dropout(DropoutRatio)(uconv2)
#     uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
#     uconv2 = residual_block(uconv2, start_neurons * 2)
#     uconv2 = residual_block(uconv2, start_neurons * 2, True)
#
#     # 101 -> 202
#     deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#     # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
#     uconv1 = concatenate([deconv1, d1])
#
#     uconv1 = Dropout(DropoutRatio)(uconv1)
#     uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
#     uconv1 = residual_block(uconv1, start_neurons * 1)
#     uconv1 = residual_block(uconv1, start_neurons * 1, True)
#
#     # uconv1 = Dropout(DropoutRatio/2)(uconv1)
#     # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
#     output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
#     output_layer = Activation('sigmoid')(output_layer_noActi)
#
#     return output_layer
# # plot_model(model, 'Xception.png', show_shapes=True)
#
#
# def BatchActivate(x):
#     x = BatchNormalization()(x)
#     # x = Lambda(lambda x: K.elu(x) + 1)(x)
#     x = Activation('relu')(x)
#     return x
#
# def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
#     x = Conv2D(filters, size, strides=strides, padding=padding)(x)
#     if activation == True:
#         x = BatchActivate(x)
#     return x
#
# def residual_block(blockInput, num_filters=16, batch_activate = False):
#     x = BatchActivate(blockInput)
#     x = convolution_block(x, num_filters, (3,3) )
#     x = convolution_block(x, num_filters, (3,3), activation=False)
#     x = Add()([x, blockInput])
#     if batch_activate:
#         x = BatchActivate(x)
#     return x
#
# input_layer = Input((202, 202, 1))
# output_layer = build_model(input_layer,64,0.5)
#
# model = Model(input_layer, output_layer)
# from keras import optimizers
#
# c = optimizers.adam(lr = 0.01)
#
# model.compile(optimizer=c, loss='binary_crossentropy')
# model.load_weights('xception.h5',by_name=True)
# print(model.get_layer(name='block2_sepconv2_act'))
# plot_model(model, 'Unet-x1.png', show_shapes=True)



