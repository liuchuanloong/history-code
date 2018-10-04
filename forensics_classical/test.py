#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:02:15 2017

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
from PIL import Image


#def get_file():
#    labelfile=open("/home/yanyu/tensorflow/example/MMF/label_3.txt") 
##    labelfile=open("/home/yanyu/tensorflow/example/MMF/test_MF3_JPEG70.txt")
#    lines=labelfile.readlines()
#    img_path = []
#    label = []
#    file_dir = os.getcwd()
#    for line in lines:
#        
#        img_path.append(file_dir +'/MMF/3/' + line.split( )[0])
##        img_path.append(file_dir +'/MMF/test_MF3_JPEG70/' + line.split( )[0].split('/')[-1])
#        label.append(int(line.split( )[1]))
#    return img_path ,label
##then I got two list which contains all testset's information
#
#def get_batch(image, label, image_W, image_H, batch_size, capacity):
#    '''
#    Args:
#        image: list type   you can feed img_path
#        label: list type    you can feed label
#        image_W: image width
#        image_H: image height
#        batch_size: batch size
#        capacity: the maximum elements in queue
#    Returns:
#        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
#        label_batch: 1D tensor [batch_size], dtype=tf.int32
#    '''
#    
#    image = tf.cast(image, tf.string)
#    label = tf.cast(label, tf.int32)
#
#    # make an input queue
#    input_queue = tf.train.slice_input_producer([image, label],shuffle = False)
#    
#    label = input_queue[1]
#    image_contents = tf.read_file(input_queue[0])
#    image = tf.image.decode_jpeg(image_contents, channels=1)
#    
#    ######################################
#    # data argumentation should go to here
#    ######################################
#    
##    image = tf.image.resize_images(image, [image_W, image_H],method = 2)
#    image = tf.reshape(image, [64, 64, 1])
#    '''
#    the way of tf.image.resize_image_with_crop_or_pad() will get a part of the picture which even human can't recognize
#    method = 0 get accuarcy(49/64)
#    method = 1 get accuarcy(48/64)
#    method = 2 get best accuarcy (50/64)
#    method = 3 get accuarcy(49/64)
#    when I test use one picture ,I got accuarcy(51/64)
#    '''
#    # if you want to test the generated batches of images, you might want to comment the following line.
#    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
#    # 训练时不要注释掉！
#    image = tf.image.per_image_standardization(image)
#    
#    image_batch, label_batch = tf.train.batch([image, label],
#                                                batch_size= batch_size,
#                                                num_threads= 1, 
#                                                capacity = capacity)
#    
#    label_batch = tf.reshape(label_batch, [batch_size])
#    image_batch = tf.cast(image_batch, tf.float32)
#    
#    return image_batch, label_batch   


def Weight_varible(shape,stddev):
    initial = tf.truncated_normal(shape,stddev=stddev)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.0,shape=shape)
    return tf.Variable(initial)
def net(x_filter,keep_prob):
    with tf.name_scope('conv1'):
        W_conv1 = Weight_varible([5,5,1,128],stddev=0.01)
        b_conv1 = bias_variable([128])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_filter,W_conv1,strides=[1,1,1,1],padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('conv2'):
    #创建卷积层conv2
        W_conv2 = Weight_varible([5,5,128,256],stddev=0.01)
        b_conv2 = bias_variable([256])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding = 'SAME')
    with tf.name_scope('conv3'):
        #创建卷积层conv3
        W_conv3 = Weight_varible([5,5,256,384],stddev=0.01)
        b_conv3 = bias_variable([384])
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv3,strides=[1,1,1,1],padding = 'SAME') + b_conv3)
    with tf.name_scope('conv4'):
        #创建卷积层conv4
        W_conv4 = Weight_varible([5,5,384,384],stddev=0.01)
        b_conv4 = bias_variable([384])
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,strides=[1,1,1,1],padding='SAME') + b_conv4)
    with tf.name_scope('conv5'):
        #创建卷积层conv5
        W_conv5 = Weight_varible([5,5,384,256],stddev=0.01)
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4,W_conv5,strides=[1,1,1,1],padding='SAME') + b_conv5)
        h_pool5 = tf.nn.max_pool(h_conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('fc1'):
        #创建全连接层fc1
        W_fc1 = Weight_varible([8*8*256,5120],stddev=0.01)
        b_fc1 = bias_variable([5120])
        h_pool5_flat = tf.reshape(h_pool5,[-1,8*8*256])
        h_fc1 = tf.matmul(h_pool5_flat,W_fc1) + b_fc1
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)
    with tf.name_scope('fc2'):
        W_fc2 = Weight_varible([5120, 5120],stddev=0.01)
        b_fc2 = bias_variable([5120])
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob=keep_prob)
    with tf.name_scope('fc3'):
        W_fc3 = Weight_varible([5120,2],stddev=0.01)
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
#def train_read_data():    
#    train_letter = '64_jpeg70'
#    train_dir='/home/liuchuanloong/amy/database_0/train/'
#    image_list,label_list=get_files(train_dir,train_letter) 
#    save_dir='/home/liuchuanloong/amy/database_0/'
#    convert_to_tfrecord(image_list,label_list,save_dir,name='0_train')
#    tfrecord_dir='/home/liuchuanloong/amy/database_0/0_train.tfrecords'
#    image_batch,label_batch=read_and_decode(tfrecord_dir,BATCH_SIZE)
#    label_batch = dense_to_one_hot(label_batch=label_batch,n_classes=n_classes,BATCH_SIZE=BATCH_SIZE)
#    return image_batch,label_batch

#def val_read_data():
#    test_letter = '64_jpeg70_test'
#    val_dir = '/home/liuchuanloong/amy/database_1/test1/'
#    val_image_list,val_label_list = get_files(val_dir,test_letter)
#    val_save_dir = '/home/liuchuanloong/amy/database_0/'    
#    convert_to_tfrecord(val_image_list,val_label_list,val_save_dir,name='0_val')
#    val_tfrecord_dir = '/home/liuchuanloong/amy/database_0/0_val.tfrecords'
#    val_image_batch,val_label_batch = read_and_decode(val_tfrecord_dir,BATCH_SIZE)
#    val_label_batch = dense_to_one_hot(label_batch=val_label_batch,n_classes=n_classes,BATCH_SIZE=BATCH_SIZE)
#    return val_image_batch,val_label_batch

    


##TRAIN
#with tf.name_scope('train_read_data'):
#    tra_image_batch,tra_label_batch = train_read_data()
#    
#    keep_prob = tf.placeholder("float")
#    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, 1])
#    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, n_classes]) 
#
#logits=net(x,keep_prob=0.5)
#with tf.name_scope('loss'):
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
#tf.summary.scalar('loss', loss)
#with tf.name_scope('train'):
#    train_op = tf.train.GradientDescentOptimizer(0.005).minimize(loss) 
#with tf.name_scope('accuracy'):
#    accuracy = evaluation(logits,y_)
#tf.summary.scalar('accurary', accuracy)
#train_log_dir = '/home/liuchuanloong/amy/database_0/3_log/'
#init=tf.global_variables_initializer()
#saver = tf.train.Saver(tf.global_variables())
#summary_op = tf.summary.merge_all() 
#with tf.Session() as sess:
#    
#    sess.run(init) 
#    coord = tf.train.Coordinator() 
#    threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
#    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
#
##    max_iter=20000 
#    MAX_STEP = 10000
#
#    iter=0 
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
#                summary_str = sess.run(summary_op,feed_dict={x:tra_images, y_:tra_labels})
#                tra_summary_writer.add_summary(summary_str, step)
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
#    log_dir = '/home/liuchuanloong/amy/database_0/3_log/'
#    n_test = 600        
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
    








#def get_one_image(train):
#     '''Randomly pick one image from training data
#     Return: ndarray
#     the 'train' is one path list
#     '''
#     n = len(train)
#     ind = np.random.randint(0, n)
##     print 'got a random number'
#     img_dir = train[ind]
# 
#     image = Image.open(img_dir)
#     plt.imshow(image)
##     image = image.resize([208, 208])
#     image = np.array(image)
#     return image

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
#    image_1 = filters.median(image_array,rectangle(5,5))
#    image_1 = np.array(image_1)
#    image_array = image_1-image_array
    
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









