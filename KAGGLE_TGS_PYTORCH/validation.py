# -*- coding: utf-8 -*-
# @Author:Liu Chuanlong
from pytorch_refinenet.pytorch_refinenet import RefineNet4Cascade
from torch.autograd import Variable
import numpy as np
import os
# from refinenet_salt import EPOCH,N_CLASS,CUDA

# IU_scores = np.zeros((EPOCH, N_CLASS))
# pixel_scores = np.zeros(EPOCH)
N_CLASS = 2

def iou(pred, target):
    ious = []
    for cls in range(2):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious

def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def val(model, val_loader, epoch, CUDA = True):
    model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if CUDA:
            inputs = Variable(batch[0].cuda())
        else:
            inputs = Variable(batch[0])

        output = model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, N_CLASS).argmax(axis=1).reshape(N, h, w)

        target = batch[2].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    # IU_scores[epoch] = ious
    # np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    # pixel_scores[epoch] = pixel_accs
    # np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)

