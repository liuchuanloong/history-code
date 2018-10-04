#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:37:12 2017

@author: liuchuanloong
"""

import tensorflow as tf
import numpy as np
import os
import string
#import  cv2 
import skimage.io as io
import matplotlib.pyplot as plt 
from skimage import filters
from skimage.morphology import rectangle

def get_files(file_dir):
    images=[]
    labels=[]
    temp=[]
    
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for name in files:
            images.append(os.path.join(root, name))
        # get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root, name))
    for one_folder in temp:        
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]
            
        if letter=='64_jpeg70_test':
            labels = np.append(labels, n_img*[0])
        else:
            labels = np.append(labels, n_img*[1])
            
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
             
    return image_list, label_list   
  

def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images, labels, save_dir,name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
    
    filename = os.path.join(save_dir,name + '.tfrecords')
    n_samples = len(labels)
    
    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))
    
    
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i]) # type(image) must be array!
            image_1 = filters.median(image,rectangle(3,3))
            image = np.array(image)
            image_1 = np.array(image_1)
            image = image_1-image
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                            'label':int64_feature(label),
                            'img_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')



def read_and_decode(filename,BATCH_SIZE):

    #根据文件名生成一个队列

    #读取已有的tfrecords，返回图片和标签

    filename_queue = tf.train.string_input_producer([filename])

 

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件

    features = tf.parse_single_example(serialized_example,

                                       features={

                                           'label': tf.FixedLenFeature([], tf.int64),

                                           'img_raw' : tf.FixedLenFeature([], tf.string),

                                       })

 

    img = tf.decode_raw(features['img_raw'], tf.uint8)

    image = tf.reshape(img, [64, 64,1])
    image = tf.cast(image, tf.float32)
#    image = tf.cast(image,tf.float32)
#    image = np.array(image)
#    print image
#    image = filters.median(image,disk(3))
    
#    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image=tf.image.per_image_standardization(image)

    label = tf.cast(features['label'], tf.int32)
    
    


    image_batch, label_batch = tf.train.shuffle_batch(
                                    [image, label], 
                                    batch_size = BATCH_SIZE,
                                    num_threads= 2,
                                    capacity = 3000,
                                    min_after_dequeue = 2000)

    
    print('Read data done!!!')
    return image_batch, label_batch


def Weight_varible(shape):
    initial = tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.0,shape=shape)
    return tf.Variable(initial)
def net(x_filter,keep_prob):
    W_conv1 = Weight_varible([5,5,1,128])
    b_conv1 = bias_variable([128])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_filter,W_conv1,strides=[1,1,1,1],padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    #创建卷积层conv2
    W_conv2 = Weight_varible([5,5,128,256])
    b_conv2 = bias_variable([256])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding = 'SAME')
    #创建卷积层conv3
    W_conv3 = Weight_varible([5,5,256,384])
    b_conv3 = bias_variable([384])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv3,strides=[1,1,1,1],padding = 'SAME') + b_conv3)
    #创建卷积层conv4
    W_conv4 = Weight_varible([5,5,384,384])
    b_conv4 = bias_variable([384])
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,strides=[1,1,1,1],padding='SAME') + b_conv4)
    #创建卷积层conv5
    W_conv5 = Weight_varible([5,5,384,256])
    b_conv5 = bias_variable([256])
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4,W_conv5,strides=[1,1,1,1],padding='SAME') + b_conv5)
    h_pool5 = tf.nn.max_pool(h_conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    #创建全连接层fc1
    W_fc1 = Weight_varible([8*8*256,5120])
    b_fc1 = bias_variable([5120])
    h_pool5_flat = tf.reshape(h_pool5,[-1,8*8*256])
    h_fc1 = tf.matmul(h_pool5_flat,W_fc1) + b_fc1
    

    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)
    
    W_fc2 = Weight_varible([5120, 5120])
    b_fc2 = bias_variable([5120])
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob=keep_prob)
    
    W_fc3 = Weight_varible([5120,2])
    b_fc3 = bias_variable([2])
    y_conv = tf.add(tf.matmul(h_fc2_drop,W_fc3) , b_fc3)
    return y_conv

def evaluation(logits,labels):
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    return accuracy

def dense_to_one_hot(label_batch,n_classes,BATCH_SIZE):
    labels = tf.one_hot(label_batch,n_classes)
    labels = tf.reshape(labels,[BATCH_SIZE,n_classes])
    return labels 
##TRAIN###################
BATCH_SIZE=128
n_classes = 2
def train_read_data():
    train_dir='/home/liuchuanloong/amy/database_1/train/'
    image_list,label_list=get_files(train_dir) 
    save_dir='/home/liuchuanloong/amy/database_1/'
    convert_to_tfrecord(image_list,label_list,save_dir,name='0_train')
    tfrecord_dir='/home/liuchuanloong/amy/database_1/0_train.tfrecords'
    image_batch,label_batch=read_and_decode(tfrecord_dir,BATCH_SIZE)
    label_batch = dense_to_one_hot(label_batch=label_batch,n_classes=n_classes,BATCH_SIZE=BATCH_SIZE)
    return image_batch,label_batch

def val_read_data():
    val_dir = '/home/liuchuanloong/amy/database_1/test2/'
    val_image_list,val_label_list = get_files(val_dir)
    val_save_dir = '/home/liuchuanloong/amy/database_1/'    
    convert_to_tfrecord(val_image_list,val_label_list,val_save_dir,name='0_val')
    val_tfrecord_dir = '/home/liuchuanloong/amy/database_1/0_val.tfrecords'
    val_image_batch,val_label_batch = read_and_decode(val_tfrecord_dir,BATCH_SIZE)
    val_label_batch = dense_to_one_hot(label_batch=val_label_batch,n_classes=n_classes,BATCH_SIZE=BATCH_SIZE)
    return val_image_batch,val_label_batch



##TRAIN
#tra_image_batch,tra_label_batch = train_read_data()
##val_image_batch,val_label_batch = val_read_data()
#
#keep_prob = tf.placeholder("float")
#x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, 1])
#y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, n_classes]) 
#
#logits=net(x,keep_prob=0.5)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
#train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss) 
#accuracy = evaluation(logits,y_)
#
#train_log_dir = '/home/liuchuanloong/amy/database_1/1_log/'
#init=tf.global_variables_initializer()
#saver = tf.train.Saver(tf.global_variables())
#with tf.Session() as sess: 
#    sess.run(init) 
#
#    coord = tf.train.Coordinator() 
#
#    threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
#
##    max_iter=20000 
#    MAX_STEP = 20000
#
#    iter=0 
#
##    if os.path.exists(os.path.join("model",'model.ckpt')) is True: 
##
##        tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model.ckpt')) 
##
##    while iter<max_iter: 
##
##        loss_np, _, acc=session.run([loss,train_optimizer,accuracy]) 
##
##        if iter%50==0: 
##            print 'step %d'%iter
##
##            print 'trainloss:',loss_np 
##            print 'accuracy:',acc
##
##        iter+=1 
##
##    coord.request_stop()#queue需要关闭，否则报错 
##
##    coord.join(threads)
#    try:
#        for step in np.arange(MAX_STEP):
#            if coord.should_stop():
#                    break
#                
#            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
#            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
#                                            feed_dict={x:tra_images, y_:tra_labels})            
#            if step % 100 == 0 or (step + 1) == MAX_STEP:                 
#                print ('Step: %d, loss: %.4f, accuracy: %.4f' % (step, tra_loss, tra_acc))
#                
##            if step % 200 == 0 or (step + 1) == MAX_STEP:
##                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
##                val_loss, val_acc = sess.run([loss, accuracy],
##                                             feed_dict={x:val_images,y_:val_labels})
##                print('**  Step %d, val loss = %.2f, val accuracy = %.2f  **' %(step, val_loss, val_acc))
#
#                    
#            if step % 1000 == 0 or (step + 1) == MAX_STEP:
#                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
#                saver.save(sess, checkpoint_path, global_step=step)
#                
#    except tf.errors.OutOfRangeError:
#        print('Done training -- epoch limit reached')
#    finally:
#        coord.request_stop()
#        
#    coord.join(threads)
#    sess.close()
    
#TEST ##########  
#def num_correct_prediction(logits, labels):
#  """Evaluate the quality of the logits at predicting the label.
#  Return:
#      the number of correct predictions
#  """
#  correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
#  correct = tf.cast(correct, tf.int32)
#  n_correct = tf.reduce_sum(correct)
#  return n_correct
#
#import math
#with tf.Graph().as_default():
#        
##        log_dir = 'C://Users//kevin//Documents//tensorflow//VGG//logsvgg//train//'
#    log_dir = '/home/liuchuanloong/amy/database_1/1_log/'
#    n_test = 7000        
#    val_image_batch,val_label_batch = val_read_data()
#
#    logits = net(val_image_batch,keep_prob=1)
#    correct = num_correct_prediction(logits, val_label_batch)
#    saver = tf.train.Saver(tf.global_variables())
#        
#    with tf.Session() as sess:
#            
#        print("Reading checkpoints...")
#        ckpt = tf.train.get_checkpoint_state(log_dir)
#        if ckpt and ckpt.model_checkpoint_path:
#            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#            saver.restore(sess, ckpt.model_checkpoint_path)
#            print('Loading success, global_step is %s' % global_step)
#        else:
#            print('No checkpoint file found')
#        
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
#            
#        try:
#            print('\nEvaluating......')
#            num_step = int(math.floor(n_test / BATCH_SIZE))
#            num_sample = num_step*BATCH_SIZE
#            step = 0
#            total_correct = 0
#            while step < num_step and not coord.should_stop():
#                batch_correct = sess.run(correct)
#                total_correct += np.sum(batch_correct)
#                step += 1
#            print('Total testing samples: %d' %num_sample)
#            print('Total correct predictions: %d' %total_correct)
#            print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
#        except Exception as e:
#            coord.request_stop(e)
#        finally:
#            coord.request_stop()
#            coord.join(threads) 
    
    
    
    
    

## TEST ###################################################### TEST #
#BATCH_SIZE=1
#train_dir='/home/liuchuanloong/amy/BOSSbase_1.01/train/'
#image_list,label_list=get_files(train_dir)  
#
#save_dir='/home/liuchuanloong/amy/BOSSbase_1.01/'    
#convert_to_tfrecord(image_list,label_list,save_dir)
#
#tfrecord_dir='/home/liuchuanloong/amy/BOSSbase_1.01/0_test.tfrecords'
#image,label=read_and_decode(tfrecord_dir,BATCH_SIZE)
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#     
#    try:
#        while not coord.should_stop() and i<1:
#             
#            img, label = sess.run([image, label])
#             
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,0],plt.cm.gray)
#                plt.show()
#            i+=1
#             
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)

from PIL import Image
def evaluate_one_image(N):
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
#    train_dir = '/home/yanyu/cat_vs_dog/train/'
#    train, train_label, combined_list = shuffle_cat_and_dog.get_files(train_dir)
    print 'get_files done'
    '''because I don't use the file of input_data.py. 
    It will be simple for me to pick a image manually  
    '''
#    image_array = get_one_image(train)
#    print 'got image done'
    cwd = os.getcwd()
#    img_dir = cwd +'/MMF/test_MF3_JPEG70/' + 'MB%s.jpeg' %N
    img_dir = cwd +'/test6/64_jpeg70_mdf3_test/' + '%s.jpeg' %N
    image_array = Image.open(img_dir)

#    image = image.resize([208, 208])
#    plt.figure()
#    plt.imshow(image_raw,cmap ='gray')
#    image.save(cwd +'/subset_train2/' + 'cat.%s.jpg' %N)
    image_array = np.array(image_array)
    image_1 = filters.median(image_array,rectangle(3,3))
    image_1 = np.array(image_1)
    image_array = image_1-image_array
    
    with tf.Graph().as_default():
#        BATCH_SIZE = 1
        
        
        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [64, 64, 1])
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 1])
        logit = net(image, keep_prob=1)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[64, 64])
        # you need to change the directories to yours.
        logs_train_dir = '/home/liuchuanloong/amy/database_1/1_log/' 
#        logs_train_dir = '/home/yanyu/tensorflow/example/logs/train/'               
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a original picture with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a filted picture with possibility %.6f' %prediction[:, 1])

for i in range(5,27):
    evaluate_one_image(i)
    print i









