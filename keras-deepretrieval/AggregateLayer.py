#-*- coding:utf-8 -*-
# Author:LIU Chuanlong
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class AggregateLayer(Layer):

    def __init__(self,output_dims,**kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.output_dims = output_dims
        super(AggregateLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[3]

    def call(self, x):
        a = K.sum(x,axis=3,keepdims=True)
        return a

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return input_shape[0], self.output_dims, 1, 1
        else:
            return(input_shape[0], 1, 1, self.output_dims)

class L2_NormalizationLayer(Layer):

    def __init__(self, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.channel_dims = 1
        super(L2_NormalizationLayer, self).__init__(**kwargs)


    def call(self, x):
        l2_normalize = tf.nn.l2_normalize(x, dim=-1, epsilon=1e-12, name='l2_norm')
        return l2_normalize

    def compute_output_shape(self,input_shape):
        return (input_shape)


