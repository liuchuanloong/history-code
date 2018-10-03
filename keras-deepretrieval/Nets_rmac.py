#-*- coding:utf-8 -*-
# Author:LIU Chuanlong
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,BatchNormalization,Reshape,Conv2D,MaxPooling2D,Lambda,TimeDistributed
from keras.optimizers import SGD,Adam
from keras.utils import np_utils,generic_utils,normalize
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from AggregateLayer import AggregateLayer,L2_NormalizationLayer
from keras import backend as K
from RoiPooling import RoiPooling

def addition(x):
    sum = K.sum(x, axis=1)
    return sum

def Classification_net(img_input,classes):

    model = VGG16(input_tensor=img_input,weights='imagenet', include_top=False)
    for i, layer in enumerate(model.layers):
       print(i, layer.name)
    x = model.output
    # x = K.batch_flatten(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(classes, activation='softmax', name='predictions')(x)
    return predictions

def Rmac_net(image_a,roi_a,num_rois):

    #create image_a model
    # Block 1
    # image_a = get_source_inputs(image_a)
    # roi_a = get_source_inputs(roi_a)
    img_a = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(image_a)
    img_a = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(img_a)
    img_a = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(img_a)

    # Block 2
    img_a = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(img_a)
    img_a = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(img_a)
    img_a = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(img_a)

    # Block 3
    img_a = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(img_a)
    img_a = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(img_a)
    img_a = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(img_a)
    img_a = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(img_a)

    # Block 4
    img_a = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(img_a)
    img_a = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(img_a)
    img_a = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(img_a)
    img_a = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(img_a)

    # Block 5
    img_a = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(img_a)
    img_a = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(img_a)
    img_a = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(img_a)
    img_a = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(img_a)

    # Get rmac regions with a RoiPooling layer.
    # If batch size was 1, we end up with N_regions x D x pooled_h x pooled_w
    # x = keras.layers.concatenate([x, rois_input])
    # num_rois = len(roi_input)
    x = RoiPooling([1], num_rois)([img_a, roi_a])
    # out_roi_pool_a = RoiPoolingConv(pool_size=1, num_rois=num_rios)([img_a, roi_a])
    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    embeding = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)
    return embeding



    # pooled_rois_centered_a = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')(out_roi_pool_a)
    # # out_roi_pool_a = L2_NormalizationLayer()(out_roi_pool_a)
    # # out_roi_pool_a = tf.nn.l2_normalize(x=out_roi_pool_a, dim=-1, epsilon=1e-12, name='l2_norm') #l2 norm
    # # out_roi_pool_a = normalize(x=out_roi_pool_a, axis=-1, order=2)
    # out_roi_pool_flat_a = Flatten()(pooled_rois_centered_a)
    #
    # # Mean center done with a scaling (at 1) + shifting.
    # # The shifting needs to be copied into the model weights. scale
    # # pooled_rois_centered_a = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')(out_roi_pool_flat_a)
    # # And then the PCA, which is another FC layer
    # pca_a = Dense(2048, kernel_initializer='RandomUniform', name='PCA')(out_roi_pool_flat_a)
    #
    # # After the FC layers the size is now n_batches x d.
    # # We reshape to n_batches x d x 1 x 1 to L2 normalize again
    # pca_reshape_a = Reshape((1,1,-1))(pca_a)
    # pca_norm_a = L2_NormalizationLayer()(pca_reshape_a)
    # # pca_norm_a = tf.nn.l2_normalize(x=pca_reshape_a, dim=-1, epsilon=1e-12, name='l2_norm') #l2 norm
    # # pca_norm_a = normalize(x=pca_reshape_a, axis=-1, order=2) #l2 norm
    # rmac_a = AggregateLayer(1)(pca_norm_a)
    # l2_rmac_a = BatchNormalization()(rmac_a)
    # rmac_flat = Flatten()(l2_rmac_a)
    # embedding = Dense(256,kernel_initializer='RandomNormal',name='embedding')(rmac_flat)












        # Vgg_model = VGG16(weights='imagenet',include_top=False)
        # for i, layer in enumerate(Vgg_model.layers):
        #    print(i, layer.name)
        # x = Vgg_model.output

        # # Block 1
        # block1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        # block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        # block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
        #
        # # Block 2
        # block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        # block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        # block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
        #
        # # Block 3
        # block3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        # block3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        # block3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        # block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
        #
        # # Block 4
        # block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        # block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        # block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        # block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
        #
        # # Block 5
        # block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        # block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        # block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        # block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
        #
        # #share PCA layer
        # pooled_rois_centered = BatchNormalization()
        # PCA = Dense(2048, kernel_initializer='RandomUniform', name='PCA')



        # #create image_p model
        # img_p = block1_conv1(image_p)
        # img_p = block1_conv2(img_p)
        # img_p = block1_pool(img_p)
        #
        # img_p = block2_conv1(img_p)
        # img_p = block2_conv2(img_p)
        # img_p = block2_pool(img_p)
        #
        # img_p = block3_conv1(img_p)
        # img_p = block3_conv2(img_p)
        # img_p = block3_conv3(img_p)
        # img_p = block3_pool(img_p)
        #
        # img_p = block4_conv1(img_p)
        # img_p = block4_conv2(img_p)
        # img_p = block4_conv3(img_p)
        # img_p = block4_pool(img_p)
        #
        # img_p = block5_conv1(img_p)
        # img_p = block5_conv2(img_p)
        # img_p = block5_conv3(img_p)
        # img_p = block5_pool(img_p)
        #
        # out_roi_pool_p = RoiPoolingConv(pooling_regions=1, num_rois=num_rios)([img_p, roi_p])
        # out_roi_pool_p = BatchNormalization()(out_roi_pool_p)  # l2 norm
        # out_roi_pool_flat_p = Flatten(2048)(out_roi_pool_p)
        #
        # pooled_rois_centered_p = pooled_rois_centered(out_roi_pool_flat_p)
        # pca_p = PCA(pooled_rois_centered_p)
        #
        # pca_reshape_p = Reshape(1, 1, None)(pca_p)
        # pca_norm_p = BatchNormalization()(pca_reshape_p)  # l2 norm
        # rmac_p = AggregateLayer()(pca_norm_p)
        # final_rmac_p = BatchNormalization()(rmac_p)
        #
        #
        #
        # #create image_n model
        # img_n = block1_conv1(image_n)
        # img_n = block1_conv2(img_n)
        # img_n = block1_pool(img_n)
        #
        # img_n = block2_conv1(img_n)
        # img_n = block2_conv2(img_n)
        # img_n = block2_pool(img_n)
        #
        # img_n = block3_conv1(img_n)
        # img_n = block3_conv2(img_n)
        # img_n = block3_conv3(img_n)
        # img_n = block3_pool(img_n)
        #
        # img_n = block4_conv1(img_n)
        # img_n = block4_conv2(img_n)
        # img_n = block4_conv3(img_n)
        # img_n = block4_pool(img_n)
        #
        # img_n = block5_conv1(img_n)
        # img_n = block5_conv2(img_n)
        # img_n = block5_conv3(img_n)
        # img_n = block5_pool(img_n)
        #
        # out_roi_pool_n = RoiPoolingConv(pooling_regions=1, num_rois=num_rios)([img_n, roi_n])
        # out_roi_pool_n = BatchNormalization()(out_roi_pool_n)  # l2 norm
        # out_roi_pool_flat_n = Flatten(2048)(out_roi_pool_n)
        #
        # pooled_rois_centered_n = pooled_rois_centered(out_roi_pool_flat_n)
        # pca_n = PCA(pooled_rois_centered_n)
        #
        # pca_reshape_n = Reshape(1, 1, None)(pca_n)
        # pca_norm_n = BatchNormalization()(pca_reshape_n)  # l2 norm
        # rmac_n = AggregateLayer()(pca_norm_n)
        # final_rmac_n = BatchNormalization()(rmac_n)
        #
        #
        # return final_rmac_a,final_rmac_p,final_rmac_n






