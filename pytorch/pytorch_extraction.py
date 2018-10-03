# -*- coding: utf-8 -*-
# @Author:Liu Chuanlong
import torch.utils.data as data
import os
from torchvision.models import vgg16
from INSTRE import INSTREclassification as INCLS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
import time


class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


def extract_feature(model, inputs):
    model.fc = torch.nn.LeakyReLU(0.1)
    model.eval()

    result = model(inputs)
    result_npy = result.data.numpy()

    return result_npy[0]


def get_features(pretrained_model, inputs):
    net1 = nn.Sequential(*list(pretrained_model.children())[0])
    out1 = net1(inputs)
    return out1


train_dataset = INCLS('/home/liuchuanloong/amy/deep_retrieval/SPN_RETRIEVAL/salbow', 'train', transforms.Compose([
    Warp(224),
    # transforms.ToTensor(),
    # transforms.Scale(224),
    transforms.RandomHorizontalFlip(),
    lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).float(),
    lambda x: x.index_select(0, torch.LongTensor([2, 1, 0]))]))

val_dataset = INCLS('/home/liuchuanloong/amy/deep_retrieval/SPN_RETRIEVAL/salbow', 'test', transforms.Compose([
    Warp(224),
    # transforms.ToTensor(),    # transforms.Scale(224),
    transforms.RandomHorizontalFlip(),
    lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).float(),
    lambda x: x.index_select(0, torch.LongTensor([2, 1, 0]))]))

# data loader
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                               num_workers=2)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False,
                             num_workers=2)

print('start!!!')
start = time.time()
VGG16 = vgg16(pretrained=True).cuda()
time_elapse = time.time() - start
print('transform to cuda done!!! elapse{}s'.format(time_elapse))

for step, (b_x, b_y) in enumerate(train_loader):
    inputs = torch.autograd.Variable(b_x[0]).cuda()
    print(get_features(VGG16, inputs))