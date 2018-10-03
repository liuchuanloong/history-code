#-*- coding:utf-8 -*-
# Author:LIU Chuanlong
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from triplet_loss import batch_all_triplet_loss


class TripletLossLayer(Layer):

    def __init__(self,**kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        super(TripletLossLayer, self).__init__(**kwargs)

    def call(self, x):
        triplet_loss = batch_all_triplet_loss(x)
        return triplet_loss

    def compute_output_shape(self, input_shape):
        return(1,)