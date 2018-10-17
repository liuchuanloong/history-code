from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
from scipy.misc import imread
import ast
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from utils import helpers

## submission accuracy ##
def calc_iou(actual, pred):
    intersection = np.count_nonzero(actual * pred)
    union = np.count_nonzero(actual + pred)
    iou_result = intersection / union if union != 0 else 0.
    return iou_result


def calc_ious(actuals, preds):
    ious_ = np.array([calc_iou(a, p) for a, p in zip(actuals, preds)])
    return ious_


def calc_precisions(thresholds, ious):
    thresholds = np.reshape(thresholds, (1, -1))
    ious = np.reshape(ious, (-1, 1))
    ps = ious > thresholds
    mps = ps.mean(axis=1)
    return mps


def indiv_scores(masks, preds):
    ious = calc_ious(masks, preds)
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = calc_precisions(thresholds, ious)

    ###### Adjust score for empty masks
    emptyMasks = np.count_nonzero(masks.reshape((len(masks), -1)), axis=1) == 0
    emptyPreds = np.count_nonzero(preds.reshape((len(preds), -1)), axis=1) == 0
    adjust = (emptyMasks == emptyPreds).astype(np.float)
    precisions[emptyMasks] = adjust[emptyMasks]
    ###################
    return precisions


def calc_metric(masks, preds):
    return np.mean(indiv_scores(masks, preds))
## submission accuracy ##

def prepare_predict_data(dataset_dir):
    pre_input_name=[]
    cwd = os.getcwd()
    for file in os.listdir(dataset_dir + "/images"):
        pre_input_name.append(cwd + "/" + dataset_dir + "/images/" + file)
    return pre_input_name

def prepare_data(dataset_dir, test = False):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    if test:
        for file in os.listdir(dataset_dir + "/test"):
            cwd = os.getcwd()
            test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
        for file in os.listdir(dataset_dir + "/test_labels"):
            cwd = os.getcwd()
            test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
        train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
        return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names
    else:
        train_input_names.sort(), train_output_names.sort(), val_input_names.sort(), val_output_names.sort()
        return train_input_names, train_output_names, val_input_names, val_output_names

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return image

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet [123.68, 116.78, 103.94]
# modified as TGS means [120.3468598, 120.3468598, 120.3468598]
def mean_image_subtraction(inputs, means=[120.3468598, 120.3468598, 120.3468598], std = [41.07161215, 41.07161215, 41.07161215]):
    # inputs=tf.to_float(inputs)
    # num_channels = inputs.get_shape().as_list()[-1]
    # if len(means) != num_channels:
    #   raise ValueError('len(means) must match the number of channels')
    # channels = tf.split(axis=2, num_or_size_splits=num_channels, value=inputs)
    # for i in range(num_channels):
    #     channels[i] -= means[i]
    #     channels[i] = channels[i]/std[i]
    # return tf.concat(axis=2, values=channels)
    num_channels = inputs.shape[2]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = np.split(inputs, 3, axis=2)
    for i in range(num_channels):
        channels[i] -= means[i]
        channels[i] = channels[i]/std[i]
    return np.concatenate(channels, axis=2)

# def _lovasz_grad(gt_sorted):
#     """
#     Computes gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     gts = tf.reduce_sum(gt_sorted)
#     intersection = gts - tf.cumsum(gt_sorted)
#     union = gts + tf.cumsum(1. - gt_sorted)
#     jaccard = 1. - intersection / union
#     jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
#     return jaccard
#
# def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
#     """
#     Flattens predictions in the batch
#     """
#     if order == 'BCHW':
#         probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
#         order = 'BHWC'
#     if order != 'BHWC':
#         raise NotImplementedError('Order {} unknown'.format(order))
#     C = probas.shape[3]
#     probas = tf.reshape(probas, (-1, C))
#     labels = tf.reshape(labels, (-1,))
#     if ignore is None:
#         return probas, labels
#     valid = tf.not_equal(labels, ignore)
#     vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
#     vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
#     return vprobas, vlabels
#
# def _lovasz_softmax_flat(probas, labels, only_present=True):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
#       labels: [P] Tensor, ground truth labels (between 0 and C - 1)
#       only_present: average only on classes present in ground truth
#     """
#     C = probas.shape[1]
#     losses = []
#     present = []
#     for c in range(C):
#         fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground for class c
#         if only_present:
#             present.append(tf.reduce_sum(fg) > 0)
#         errors = tf.abs(fg - probas[:, c])
#         errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
#         fg_sorted = tf.gather(fg, perm)
#         grad = _lovasz_grad(fg_sorted)
#         losses.append(
#             tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
#                       )
#     losses_tensor = tf.stack(losses)
#     if only_present:
#         present = tf.stack(present)
#         losses_tensor = tf.boolean_mask(losses_tensor, present)
#     return losses_tensor
#
# def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
#       labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
#       only_present: average only on classes present in ground truth
#       per_image: compute the loss per image instead of per batch
#       ignore: void class labels
#       order: use BHWC or BCHW
#     """
#     probas = tf.nn.softmax(probas)
#     labels = helpers.reverse_one_hot(labels)
#
#     if per_image:
#         def treat_image(prob, lab):
#             prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
#             prob, lab = _flatten_probas(prob, lab, ignore, order)
#             return _lovasz_softmax_flat(prob, lab, only_present=only_present)
#         losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
#     else:
#         losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
#     return losses

# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=False, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true, -1), 'int32')
    y_pred = tf.cast(tf.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = False, ignore = None)
    return loss


# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def mean_score(y_true, y_pred):
    """
    Calculate mean score for batch images

    :param y_true: 4-D Tensor of ground truth, such as [NHWC]. Should have numeric or boolean type.
    :param y_pred: 4-D Tensor of prediction, such as [NHWC]. Should have numeric or boolean type.
    :return: 0-D Tensor of score
    """
    y_true_ = tf.cast(tf.round(y_true), tf.bool)
    y_pred_ = tf.cast(tf.round(y_pred), tf.bool)

    # Flatten
    y_true_ = tf.reshape(y_true_, shape=[tf.shape(y_true_)[0], -1])
    y_pred_ = tf.reshape(y_pred_, shape=[tf.shape(y_pred_)[0], -1])
    threasholds_iou = tf.constant(np.arange(0.5, 1.0, 0.05), dtype=tf.float32)

    def _mean_score(y):
        """Calculate score per image"""
        y0, y1 = y[0], y[1]
        total_cm = tf.confusion_matrix(y0, y1, num_classes=2)
        total_cm = tf.Print(total_cm, [total_cm])
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag
        denominator = tf.where(tf.greater(denominator, 0), denominator, tf.ones_like(denominator))
        # iou[0]: IoU of Background
        # iou[1]: IoU of Foreground
        iou = tf.div(cm_diag, denominator)
        iou_fg = iou[1]
        greater = tf.greater(iou_fg, threasholds_iou)
        score_per_image = tf.reduce_mean(tf.cast(greater, tf.float32))
        # Both predicted object and ground truth are empty, score is 1.
        score_per_image = tf.where(
            tf.logical_and(
                tf.equal(tf.reduce_any(y0), False), tf.equal(tf.reduce_any(y1), False)),
            1., score_per_image)
        return score_per_image

    elems = (y_true_, y_pred_)
    scores_per_image = tf.map_fn(_mean_score, elems, dtype=tf.float32)
    return tf.reduce_mean(scores_per_image)

def get_iou_vector(y_pred, y_true):
    metric = []
    t, p = y_true, y_pred
    if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
        metric.append(0)
    elif np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
        metric.append(0)
    elif np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
        metric.append(1)
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    thresholds = np.arange(0.5, 1, 0.05)
    s = []
    for thresh in thresholds:
        s.append(iou > thresh)
    metric.append(np.mean(s))

    return np.mean(metric)

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

    
def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

            
        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)

