from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess


from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train for')
parser.add_argument('--checkpoint_step', type=int, default=3, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--padding', type=str2bool, default=True, help='Whether to pad the image for data augmentation')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=0.1, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="custom", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet50", help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--tensorboard', type=str, default="kaggle/log", help='The log directory tensorboard using. Specifies the tensorboard log directory')
args = parser.parse_args()


def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)
    if args.padding:
        input_image = cv2.copyMakeBorder(input_image, 16, 16, 16, 16, cv2.BORDER_REFLECT)
        output_image = cv2.copyMakeBorder(output_image, 16, 16, 16, 16, cv2.BORDER_REFLECT)
    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
    return input_image, output_image

# Prepare log dirs for tensorboard call
train_log_dir = args.tensorboard + '/' + args.model + '/train'
val_log_dir = args.tensorboard + '/' + args.model + '/val'
if not os.path.exists(train_log_dir):
    os.makedirs(train_log_dir)
if not os.path.exists(val_log_dir):
    os.makedirs(val_log_dir)

# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# addition code for tensorboard callback
def write_log(writer, logs, step, names='loss'):
    ## plot loss with tensorboard ##
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = logs
    summary_value.tag = names
    writer.add_summary(summary, step)
    writer.flush()
# addition end #
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

# Compute your softmax cross entropy loss
with tf.name_scope('input'):
    net_input = tf.placeholder(tf.float32,shape=[None,128,128,3])
    net_output = tf.placeholder(tf.float32,shape=[None,128,128,1])
if args.padding:
    with tf.name_scope('network'):
        network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes-1, crop_width=args.crop_width+32, crop_height=args.crop_height+32, is_training=True)
else:
    with tf.name_scope('network'):
        network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes-1, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)
with tf.name_scope('loss'):
    # loss = utils.lovasz_loss(y_pred=network, y_true=net_output)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=net_output, logits=network))
with tf.name_scope('optimizer'):
    opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, var_list=[var for var in tf.trainable_variables()])

tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
val_summary_writer = tf.summary.FileWriter(val_log_dir)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = "kaggle/checkpoints/" + args.model + "/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
best_model_checkpoint_name = "kaggle/checkpoints/" + args.model + "/best_model_" + args.model + "_" + args.dataset + ".ckpt"

if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)
    length = len(model_checkpoint_name.split('/')[-1]) + len(model_checkpoint_name.split('/')[-2]) + 1
    checkpoint_epochlist = os.listdir(model_checkpoint_name[:-length])
    del checkpoint_epochlist[-1]
    del checkpoint_epochlist[-1]
    for strs in checkpoint_epochlist:
        last = int(strs)
        if last <= int(strs):
            last = int(strs)
    begin = last + 1
else:
    begin = 0
# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names = utils.prepare_data(dataset_dir=args.dataset)



print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []
avg_subacc_per_epoch = []
# # Which validation images do we want
# val_indices = []
# num_vals = min(args.num_val_images, len(val_input_names))
#
# # Set random seed to make sure models are validated on the same validation images.
# # So you can compare the results of different models more intuitively.
# random.seed(16)
# val_indices=random.sample(range(0,len(val_input_names)),num_vals)
val_indices=range(0,len(val_input_names))
subacc_tmp = 0
# Do the training here
for epoch in range(begin, args.num_epochs):

    current_losses = []

    cnt=0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st = time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            output_image = utils.load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)


                # Prep the data. Make sure the labels are in one-hot format
                # input_image = np.float32(input_image) / 255.0
                input_image = utils.mean_image_subtraction(np.float32(input_image))
                output_image = np.float32(helpers.reverse_one_hot(helpers.one_hot_it(label=output_image, label_values=label_values)))
                output_image = np.expand_dims(output_image, axis=-1)

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1), axis=0)
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1), axis=0)

        # Do the training
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 32 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()
    # Addition summary graph
    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)
    write_log(tra_summary_writer, logs=mean_loss, step=epoch)
    print("\nAverage train loss for epoch # %04d = %f" % (epoch, mean_loss))
    # Addition end

    # Create directories if needed
    if not os.path.isdir("%s/%s/%04d"%("kaggle/checkpoints",args.model,epoch)):
        os.makedirs("%s/%s/%04d"%("kaggle/checkpoints",args.model,epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%s/%04d/model.ckpt"%("kaggle/checkpoints",args.model,epoch))


    if epoch % args.validation_step == 0:
        print("Performing validation")
        target=open("%s/%s/%04d/val_scores.csv"%("kaggle/checkpoints",args.model,epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))


        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
        current_vlosses = []
        submission_acc_list = []


        # Do the validation on a small set of validation images
        for ind in val_indices:
            
            # input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
            # gt = utils.load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
            # gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
            #
            # # st = time.time()
            #
            # # addition code for loss and tensorboard show curve
            # val_input_image = utils.load_image(val_input_names[ind])
            # val_output_image = utils.load_image(val_output_names[ind])
            # val_input_image, val_output_image = utils.random_crop(val_input_image, val_output_image, args.crop_height, args.crop_width)
            #
            # val_input_image = np.expand_dims((np.float32(val_input_image) / 255.0), axis=0)
            # val_output_image = np.expand_dims(np.float32(helpers.one_hot_it(label=val_output_image, label_values=label_values)),axis=0)
            #
            # val_loss = sess.run([loss], feed_dict={net_input: val_input_image, net_output: val_output_image})
            # current_vlosses.append(val_loss)
            #
            # ################### addition end ##########################

            # Do the validation on a small set of validation images with padding
            input_image = np.expand_dims(np.float32(cv2.copyMakeBorder(utils.load_image(val_input_names[ind]), 14, 13, 14, 13, cv2.BORDER_REFLECT)),axis=0)/255.0
            gt = cv2.copyMakeBorder(utils.load_image(val_output_names[ind]), 14, 13, 14, 13, cv2.BORDER_REFLECT)
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

            # st = time.time()

            # addition code for loss and tensorboard show curve
            val_input_image = cv2.copyMakeBorder(utils.load_image(val_input_names[ind]), 14, 13, 14, 13, cv2.BORDER_REFLECT)
            val_output_image = cv2.copyMakeBorder(utils.load_image(val_output_names[ind]), 14, 13, 14, 13, cv2.BORDER_REFLECT)
            # val_input_image, val_output_image = utils.random_crop(val_input_image, val_output_image, args.crop_height, args.crop_width)

            # val_input_image = np.expand_dims((np.float32(val_input_image) / 255.0), axis=0)
            val_input_image = np.expand_dims((utils.mean_image_subtraction(np.float32(val_input_image))), axis=0)
            val_output_image = np.expand_dims(np.float32(helpers.reverse_one_hot(helpers.one_hot_it(label=val_output_image, label_values=label_values))),axis=-1)
            val_output_image = np.expand_dims(val_output_image, axis=0)

            val_loss = sess.run([loss], feed_dict={net_input: val_input_image, net_output: val_output_image})
            current_vlosses.append(val_loss)

            ################### addition end ##########################

            output_image = sess.run(network,feed_dict={net_input:input_image})

            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)
            # submission_acc = sess.run(utils.mean_score(y_true=gt, y_pred=output_image))
            # submission_acc = utils.get_iou_vector(y_true=gt[14:-13,14:-13], y_pred=output_image[14:-13,14:-13])
            submission_acc = utils.calc_metric(masks=gt[14:-13,14:-13], preds=output_image[14:-13,14:-13])

            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)
            submission_acc_list.append(submission_acc)
            
            gt = helpers.colour_code_segmentation(gt, label_values)
 
            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%s/%04d/%s_pred.png"%("kaggle/checkpoints",args.model,epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%s/%04d/%s_gt.png"%("kaggle/checkpoints",args.model,epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


        target.close()
        mean_vloss = np.mean(current_vlosses)
        write_log(val_summary_writer, logs=mean_vloss, step=epoch)
        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)
        avg_subacc = np.mean(submission_acc_list)
        avg_subacc_per_epoch.append(avg_subacc)
        write_log(val_summary_writer, logs=avg_subacc, step=epoch, names='subacc')
        if avg_subacc > subacc_tmp:
            subacc_tmp = avg_subacc
            print("Saving best checkpoint")
            saver.save(sess, best_model_checkpoint_name)

        print("\nAverage submission accuracy for epoch # %04d = %f"% (epoch, avg_subacc))
        print("Average validation loss for epoch # %04d = %f"% (epoch, mean_vloss))
        print("Average validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []


    fig1, ax1 = plt.subplots(figsize=(11, 8))

    ax1.plot(range(epoch-begin+1), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig('accuracy_vs_epochs.png')

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(range(epoch-begin+1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs.png')

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch-begin+1), avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")

    plt.savefig('iou_vs_epochs.png')

    plt.clf()

    fig4, ax4 = plt.subplots(figsize=(11, 8))

    ax4.plot(range(epoch-begin+1), avg_subacc_per_epoch)
    ax4.set_title("Average subacc vs epochs")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Current subacc")

    plt.savefig('subacc_vs_epochs.png')
