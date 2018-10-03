#-*- coding:utf-8 -*-
# Author:LIU Chuanlong
import cv2
import numpy as np
import os
from PIL import  Image

class ImageHelper:
    def __init__(self, batch_size, S=800, L=2):
        self.S = S
        self.L = L
        self.means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]
        self.index = 0
        self.batch_size = batch_size
        self.X = []
        self.Y = []

    def oneImageLable(self,image, label):
        image = cv2.imread(image)
        image = cv2.resize(image, (224, 224))
        return image, label

    def generate_one_from_list(self,temp):
        self.X = []
        self.Y = []
        self.R = []
        for n in range(self.batch_size):
            image,label = temp[self.index]
            img, reg, label = self.prepare_image_and_grid_regions_for_network(image,label)

            self.X.append(img)
            self.R.append(reg)
            self.Y.append(label)
            self.index += 1
            self.index = 0 if self.index + self.batch_size > len(temp[:,0]) else self.index
        return np.array(self.X), np.array(self.Y)

    def prepare_image_and_grid_regions_for_network(self, fname, roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        I, im_resized = self.load_and_prepare_image(fname, roi)
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = []
            all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
            R = self.pack_regions_for_network(all_regions)
        return im_resized, R

    def get_rmac_features(self, I, R, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end='rmac/normalized')
        return np.squeeze(net.blobs['rmac/normalized'].data)

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S)/np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means
        return I, im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
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

def load_and_prepare_image(fname, roi=None):
    # Read image, get aspect ratio, and resize such as the largest side equals S
    means = np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :, None, None]
    im = cv2.imread(fname)
    im_size_hw = np.array(im.shape[0:2])
    ratio = float(800) / np.max(im_size_hw)
    new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
    im_resized = cv2.resize(im, (new_size[1], new_size[0]))
    # If there is a roi, adapt the roi to the new size and crop. Do not rescale
    # the image once again
    if roi is not None:
        roi = np.round(roi * ratio).astype(np.int32)
        im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
    # Transpose for network and subtract mean
    I = im_resized.transpose(2, 0, 1) - means
    return I, im_resized
lab_path = '/home/liuchuanloong/amy/deep_retrieval/annotations_landmarks/landmarkdataset/annotations_landmarks_val_xy.txt'
IH = ImageHelper(32)

with open(lab_path, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline().strip()
        if not lines:
            break
        name_tmp, x1, y1, x2, y2, class_tmp = [temp for temp in lines.split(',')]
        x = int(x1)
        y = int(y1)
        w = int(x2) - int(x1)
        h = int(y2) - int(y1)
        roi = np.array([x, y, w, h])
        filename = os.path.join('/home/liuchuanloong/amy/deep_retrieval/annotations_landmarks/landmarkdataset',name_tmp)
        I, R = IH.prepare_image_and_grid_regions_for_network(filename)

        # img = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(np.uint8(I), cv2.COLOR_GRAY2RGB)
        cv2.imshow('re.jpeg', I)
        cv2.waitKey()



