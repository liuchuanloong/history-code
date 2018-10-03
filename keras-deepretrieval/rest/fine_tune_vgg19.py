# -*- coding: utf-8 -*-
# Author: liuchuanloong.name
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.utils import plot_model
import tensorflow as tf


train_data_dir = "/home/liuchuanloong/amy/database/256_ObjectCategories/train"
validation_data_dir = "/home/liuchuanloong/amy/database/256_ObjectCategories/test"
img_height = 224
img_width = 224
train_datagen = ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')
validation_datagen = ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True)
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')
model = VGG19(weights='imagenet', include_top=False)
for i, layer in enumerate(model.layers):
   print(i, layer.name)
x = model.output
# layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
# x = Flatten(name='flatten')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(1000, activation='relu', name='fc2')(x)
# x = Dense(classes, activation='softmax', name='predictions')(x)
# x = GlobalAveragePooling2D()(x)
# x = Dense(1000, activation='relu', name='fc1')(x)
# x = Dropout(0.5)(x)
predictions = Dense(257, activation='softmax',name='predictions')(x)
new_model = Model(inputs=model.input,outputs=predictions)

for layer in new_model.layers[:21]:
    layer.trainable = False
new_model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
new_model.summary()
plot_model(new_model, to_file='model.png')
log_filepath = './tmp/log'
filepath="./tmp/log/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
TB = TensorBoard(log_dir=log_filepath, write_images=0, histogram_freq=1)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
new_model.fit_generator(
            train_generator,validation_data=validation_generator,
            epochs=50,
            steps_per_epoch=672,
            validation_steps=284,
            callbacks=[TB,checkpoint],
            )

new_model.save('fine_tune_fc.h5')



# cbks = [tb_cb]
# history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#                     verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))
