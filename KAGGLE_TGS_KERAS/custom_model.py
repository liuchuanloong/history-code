from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from builders import frontend_builder
from keras.engine.topology import Layer

class UNetLayer(Layer):
    def __init__(self,num_classes, frontend="ResNet101", is_training=True, n_filters=32, dropout_p=0.2, **kwargs):
        self.num_classes = num_classes
        self.frontend = frontend
        self.is_training = is_training
        self.n_filters = n_filters
        self.dropout_p = dropout_p

        super(UNetLayer, self).__init__(**kwargs)

    def conv_block(self, inputs, n_filters, filter_size=[2, 2], dropout_p=0.2):
        """
        Basic conv block for Encoder-Decoder
        Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
        Dropout (if dropout_p > 0) on the inputs
        """
        conv = slim.conv2d(inputs, n_filters, filter_size, activation_fn=None, normalizer_fn=None, padding='same')
        out = tf.nn.relu(slim.batch_norm(conv, fused=True))
        if dropout_p != 0.0:
            out = slim.dropout(out, keep_prob=(1.0-dropout_p))
        return out

    def conv_transpose_block(self, inputs, middle_channels, out_channels, strides=[2,2], filter_size=[2, 2], dropout_p=0.2):
        """
        Basic conv transpose block for Encoder-Decoder upsampling
        Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
        Dropout (if dropout_p > 0) on the inputs
        """

        conv = self.conv_block(inputs, middle_channels, dropout_p=dropout_p)
        conv = slim.conv2d_transpose(conv, out_channels, kernel_size=filter_size, stride=strides, padding='same')
        out = tf.nn.relu(slim.batch_norm(conv, fused=True))
        if dropout_p != 0.0:
            out = slim.dropout(out, keep_prob=(1.0-dropout_p))
        return out

    def call(self, inputs):

        # conv = conv_block(inputs, n_filters=3, dropout_p=dropout_p)
        # conv1 = slim.max_pool2d(conv, [2,2])

        n_filters = self.n_filters
        dropout_p = self.dropout_p

        logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, self.frontend, is_training=self.is_training)

        conv2 = end_points["pool2"] # 32
        conv3 = end_points["pool3"] # 16
        conv4 = end_points["pool4"] # 8
        conv5 = end_points["pool5"] # 4

        pool = slim.max_pool2d(conv5, [2,2]) # 2
        center = self.conv_transpose_block(pool, n_filters*8*2, n_filters*8, dropout_p=dropout_p) # 4

        dec5 = self.conv_transpose_block(tf.concat([center, conv5], axis=3), n_filters*8*2, n_filters*8, dropout_p=dropout_p) # 8
        dec4 = self.conv_transpose_block(tf.concat([dec5, conv4], axis=3), n_filters*8*2, n_filters*8, dropout_p=dropout_p) # 16
        dec3 = self.conv_transpose_block(tf.concat([dec4, conv3], axis=3), n_filters*4*2, n_filters*2,dropout_p=dropout_p) # 32
        dec2 = self.conv_transpose_block(tf.concat([dec3, conv2], axis=3), n_filters*2*2, n_filters*2*2, dropout_p=dropout_p) # 64
        dec1 = self.conv_transpose_block(dec2, n_filters*2*2, n_filters,dropout_p=dropout_p) # 128
        net = slim.conv2d(dec1, self.num_classes, [1, 1])
        # if dropout_p != 0.0:
        #     net = slim.dropout(net, keep_prob=(1.0-dropout_p))
        return net