# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# import os
# import shutil
# import numpy as np
# import argparse
# from keras.preprocessing.image import load_img
# from tqdm import tqdm_notebook
#
# train_df = pd.read_csv("./data/train.csv", index_col="id", usecols=[0])
# depths_df = pd.read_csv("./data/depths.csv", index_col="id")
# train_df = train_df.join(depths_df)
# test_df = depths_df[~depths_df.index.isin(train_df.index)]
# length = np.ones((4000,1))
#
# train_length = pd.DataFrame(np.ones((3600,1)), columns=['flag'])
# test_length = pd.DataFrame(np.zeros((400,1)), columns=['flag'])
# skf = StratifiedKFold(n_splits = 10,random_state=0,shuffle=False)
# shutil.rmtree('./CV')
# os.makedirs('./CV')
# for i, (train_index, test_index) in enumerate(skf.split(train_df,length)):
#
#     train_ids, test_ids = train_df.iloc[train_index], train_df.iloc[test_index]
#     train_ids = train_ids.reset_index()
#     test_ids = test_ids.reset_index()
#
#     train_data = pd.merge(train_ids, train_length, on=train_ids.index)
#     test_data = pd.merge(test_ids, test_length, on=test_ids.index)
#
#     data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
#
#     data.to_csv('./CV/fold_%d'%(i))
#     print('**** Running fold '+ str(i))
#
#
# # Loading of training/testing ids and depths
# train_df = pd.read_csv("CV/fold_0")
#
# DATA = {'train':{},'valid':{}}
#
# DATA["train"]["images"] = [np.array(load_img("./data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df[train_df.flag==1.0].id)]
# DATA["train"]["masks"] = [np.array(load_img("./data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df[train_df.flag==1.0].id)]
# DATA["valid"]["images"] = [np.array(load_img("./data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df[train_df.flag==0.0].id)]
# DATA["valid"]["masks"] = [np.array(load_img("./data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df[train_df.flag==0.0].id)]
# x_train, y_train, x_valid, y_valid = DATA["train"]["images"], DATA["train"]["masks"], DATA["valid"]["images"], DATA["valid"]["masks"]
#


'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''


# -*- coding: utf-8 -*-
# @Author:Liu Chuanlong
import csv
import os
import os.path
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import demo.experiment.config as C

def write_object_labels_csv(file, labeled_data, classses):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(C.object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(len(classses)):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)
    csvfile.close()

def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

class INSTREclassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):

        assert(set in ['train', 'test'])
        self.root = root
        self.datapath = os.path.join(root, 'INSTRE_release', 'INSTRE-S1')
        self.datalist = os.listdir(self.datapath)
        self.label = sorted(self.datalist)
        self.labeldata = {}
        self.transform = transform
        self.target_transform = target_transform
        self.classes = C.object_categories

        for name in self.datalist:
            one = -np.ones(len(self.label))
            n = self.label.index(name)
            one[n] = 1.0
            tmppath = os.path.join(self.datapath, name)
            imagepath = os.listdir(tmppath)
            for image in imagepath:
                if image.split('.')[-1] == 'txt':
                    continue
                if set == 'train' and int(image.split('.')[-2]) < 80:
                    enimg = os.path.join(name, image)
                    self.labeldata[enimg] = one
                elif set == 'test' and int(image.split('.')[-2]) > 80:
                    enimg = os.path.join(name, image)
                    self.labeldata[enimg] = one

        # define path of csv file
        path_csv = os.path.join(self.root, 'SpnRetrievalPytorch', 'DATAFILES')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'INSTREclassification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # write csv file
            write_object_labels_csv(file_csv, self.labeldata, classses=self.classes)

        self.images = read_object_labels_csv(file_csv)

        print('[dataset] INSTRE classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]

        img = Image.open(os.path.join(self.datapath, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)



from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    for batch in datagen.flow(x_train, y_train,batch_size=batch_size):
        pass
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

