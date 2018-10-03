# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from Nets_rmac import Rmac_net,Classification_net
from keras.layers import Input,Merge,merge,concatenate,Lambda
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.utils import plot_model,np_utils
from keras import backend as K
from triplet_loss import batch_all_triplet_loss
from TripletLossLayer import TripletLossLayer
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import tensorflow as tf
import time

class Dataset:
    def __init__(self):
        self.list_image = []
        self.list_label = []

    def predata(self,root,annotationfile):
        with open(os.path.join(root,annotationfile), 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                conterpartpath, x1, y1, x2, y2 ,classes = [temp for temp in lines.split(',')]
                imagepath = os.path.join(root,conterpartpath)
                if not os.path.exists(imagepath):
                    continue
                self.list_image.append(imagepath)
                self.list_label.append(classes)
            temp = np.array([self.list_image, self.list_label])
            temp = temp.transpose()
            np.random.shuffle(temp)
            return temp

class ImageHelper:
    def __init__(self, S=800,L=3):
        self.S = S
        self.L = L
        self.index = 0


    def generate_one_from_list(self,temp,batch_size):
        self.X = []
        self.Y = []
        self.R = []
        self.batch_size = batch_size

        for n in range(self.batch_size):
            image,label = temp[self.index]
            img, reg, label = self.prepare_image_and_grid_regions_for_network(image,label)

            self.X.append(img)
            self.R.append(reg)
            self.Y.append(label)
            self.index += 1
            self.index = 0 if self.index + self.batch_size > len(temp[:,0]) else self.index
        return np.asarray(self.X), np.asarray(self.R), np.asarray(self.Y)

    def prepare_image_and_grid_regions_for_network(self, fname, label):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        im_resized = self.load_and_prepare_image(fname)
        # Get the region coordinates and feed them to the network.
        # print(label)
        # cv2.imshow('x.jpeg',im_resized)
        # cv2.waitKey()
        all_regions = []
        all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
        R = self.pack_regions_for_network(all_regions)
        return im_resized, R ,label

    def get_rmac_features(self, I, R, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end='rmac/normalized')
        return np.squeeze(net.blobs['rmac/normalized'].data)

    def random_crop(self,width, height, image):
        """Crops a random region of width x height from image.
        Returns an image"""
        crop_origin_x = randint(0, image.size[0] - width)
        crop_origin_y = randint(0, image.size[1] - height)
        crop_box = (crop_origin_x, \
                    crop_origin_y, \
                    crop_origin_x + width, \
                    crop_origin_y + height)
        return image.crop(crop_box)

    def flip_axis(self,x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    def load_and_prepare_image(self, fname):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im, (224, 224))
        # Transpose for network and subtract mean
        im_resized = np.array(im_resized, np.dtype(float)) * (1. / 255) - 0.5
        if np.random.random() < 0.5:
            im_resized = self.flip_axis(x=im_resized, axis = 1)
        return im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        # R = np.zeros((n_regs, 5), dtype=np.float32)
        R = np.zeros((n_regs, 4), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                # R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], :] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        R = np.divide(R,32)
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        # R[:n_regs, 2] = R[:n_regs, 0] + R[:n_regs, 2] - 1
        # R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        return R

    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)
class config:
    def __init__(self):
        self.root = '/home/liuchuanloong/amy/deep_retrieval/annotations_landmarks/landmarkdataset'
        self.anns = 'landmark_xy.txt'
        self.classes = 6
        self.num_rios = 5
        self.classifier =False
        self.batch_size = 8
        self.n_epochs = 2
        self.squared = False
        self.loss = []
C = config()
IMAGE = Dataset()
IMAGEHELPER = ImageHelper()
temp = IMAGE.predata(root=C.root, annotationfile=C.anns)

#################prepare nets#####################
input_shape_img = (224, 224, 3)
img_input = Input(shape=input_shape_img)
image_a = Input(shape=input_shape_img)
label = Input(shape=(1,))
roi_a = Input(shape=(C.num_rios, 4))

## plot loss ##
def write_log(callback, names, logs, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = logs
    summary_value.tag = names
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

if C.classifier:
    ###     classifier     ###
    predictions = Classification_net(img_input,C.classes)
    classification_model = Model(inputs=img_input,outputs=predictions)
    classification_model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
    classification_model.summary()
    plot_model(classification_model, to_file='classifier_model.png')

else:
    ###     rmac     ###
    embedding = Rmac_net(image_a,roi_a,C.num_rios)

    # triplet_losses = merge([label, final_rmac_a],
    #     mode=batch_all_triplet_loss,
    #     name='loss',
    #     output_shape=(1,))
    triplet_losses = Lambda(batch_all_triplet_loss)([embedding,label])
    rmac_model = Model(
        inputs=[image_a, roi_a, label],
        outputs=triplet_losses)
    rmac_model.compile(loss=identity_loss, optimizer=SGD(lr=1e-3,momentum=0.9,decay=5*1e-5))

    log_path = './log'
    callback = TensorBoard(log_path)
    callback.set_model(rmac_model)
    train_names = 'train_loss'
    # loss = TripletLossLayer()([final_rmac_a,label])
    # rmac_model = Model(inputs=[image_a,roi_a,label],outputs=loss)
    # rmac_model.compile(loss=None, optimizer=Adam(lr=1e-4))

    rmac_model.summary()
    plot_model(rmac_model, to_file='./log/rmac_model.png',show_shapes=True)

    for e in range(C.n_epochs):
        print('Training... \nEpoch', e)

        for n in range(int(temp.shape[0] / C.batch_size)):
            I, R, Y = IMAGEHELPER.generate_one_from_list(temp=temp,batch_size=C.batch_size)
            Y_batch = np_utils.to_categorical(Y, C.classes)
            rmac_loss = rmac_model.train_on_batch([I, R, Y], Y_batch)
            C.loss.append(rmac_loss)
            print('Epoch:{} batch:{}/{} train_loss:{}'
                .format(e, n, int(temp.shape[0] / C.batch_size), C.loss[n*(e+1)]))
            write_log(callback, train_names, rmac_loss, len(C.loss))
        rmac_model.save_weights('./tmp/siamesenet epoch {} {}'.format(e,time.ctime()))
    rmac_model.save_weights('./tmp/siamesenet {}'.format(time.ctime()))


