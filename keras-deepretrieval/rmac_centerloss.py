# -*- coding: utf-8 -*-
# @Author:Liu Chuanlong
import numpy as np
import cv2
import os
from keras.applications.vgg16 import VGG16
from Nets_rmac import Rmac_net
from keras.layers import Convolution2D,Lambda,BatchNormalization,Activation,GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.utils import plot_model,np_utils
from keras import backend as K
from triplet_loss import batch_all_triplet_loss
from keras.callbacks import TensorBoard
import tensorflow as tf
import time

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
        self.LAMBDA = 0.5
        self.CENTER_LOSS_ALPHA = 0.5

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
        R = np.zeros((n_regs, 4), dtype=np.float32)
        cnt = 0
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], :] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        R = np.divide(R,32)
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


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

def ClassificerNet(image):
    ##prepare nets##

    model = VGG16(input_tensor=image,weights='imagenet', include_top=False)

    x = model.output
    feature = Convolution2D(1024, (1, 1), padding='same', name='add_conv',kernel_initializer='TruncatedNormal')(x)
    x = BatchNormalization(name='add_normalization')(feature)
    x = Activation('relu',name='add_activation')(x)
    x = GlobalAveragePooling2D(name='GlobalAvgPooling')(x)
    x = Dense(2048, activation='relu', name='add_fc1',kernel_initializer='TruncatedNormal')(x)
    predictions = Dense(C.classes, activation='softmax', name='predictions',kernel_initializer='TruncatedNormal')(x)
    return feature, predictions

def write_log(callback, names, logs, batch_no):
    ## plot loss ##

    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = logs
    summary_value.tag = names
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

def build_network(images,labels,center_loss_alpha,num_classes,ratio=0.5):

    features, logits = ClassificerNet(images)

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers, centers_update_op = get_center_loss(features, labels, center_loss_alpha,
                                                                      num_classes)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * center_loss

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)

    return logits, features, total_loss, accuracy, centers_update_op


C = config()
IMAGE = Dataset()
IMAGEHELPER = ImageHelper()
temp = IMAGE.predata(root=C.root, annotationfile=C.anns)

## define placeholder for networks##
images = tf.placeholder(tf.float32, shape=[C.batch_size, 224, 224, 3])
labels = tf.placeholder(tf.float32, shape=[C.batch_size,])
global_step = tf.Variable(0, trainable=False, name='global_step')

logits, features, total_loss, accuracy, centers_update_op = build_network(images=images,
                                                                          labels=labels,
                                                                          center_loss_alpha=C.CENTER_LOSS_ALPHA,
                                                                          num_classes=C.classes ,
                                                                          ratio=C.LAMBDA)
optimizer = tf.train.AdamOptimizer(0.001)
with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)
summary_op = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('/tmp/center_loss_log', sess.graph)

step = sess.run(global_step)
while step <= 8000:
    batch_images, R, Y = IMAGEHELPER.generate_one_from_list(temp=temp, batch_size=C.batch_size)
    batch_labels = np_utils.to_categorical(Y, C.classes)
    _, summary_str, train_acc = sess.run(
        [train_op, summary_op, accuracy],
        feed_dict={
            images: batch_images,
            labels: batch_labels,
        })
    step += 1

    writer.add_summary(summary_str, global_step=step)

    print(("step: {}, train_acc:{:.4f}".
           format(step, train_acc)))