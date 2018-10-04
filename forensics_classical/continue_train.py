#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:43:26 2017

@author: liuchuanloong
"""

##CONTINUE_TRAIN###################
#
def train_read_data():    
    train_letter = '64_jpeg70'
    train_dir='/home/liuchuanloong/amy/forensics/dataset/train_1/'
    image_list,label_list=get_files(train_dir,train_letter) 
    save_dir='/home/liuchuanloong/amy/forensics/dataset/'
    convert_to_tfrecord(image_list,label_list,save_dir,name='0_train')
    tfrecord_dir='/home/liuchuanloong/amy/forensics/dataset/0_train.tfrecords'
    image_batch,label_batch=read_and_decode(tfrecord_dir,BATCH_SIZE)
    label_batch = dense_to_one_hot(label_batch=label_batch,n_classes=n_classes,BATCH_SIZE=BATCH_SIZE)
    return image_batch,label_batch

with tf.name_scope('train_read_data'):
    tra_image_batch,tra_label_batch = train_read_data()
    
    keep_prob = tf.placeholder("float")
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, 1])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, n_classes]) 

logits=net(x,keep_prob=0.5)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step=global_step,decay_steps=800,
    decay_rate=0.96,staircase=True)
    train_op = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss,global_step=global_step)
#    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step) 
with tf.name_scope('accuracy'):
    accuracy = evaluation(logits,y_)
tf.summary.scalar('accurary', accuracy)
train_log_dir = './11_log/'
#init=tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
summary_op = tf.summary.merge_all() 
with tf.Session() as sess:
    
#    sess.run(init) 
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

#    max_iter=20000 
    MAX_STEP = 10000
    
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(train_log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
    con_step = global_step + 1
    try:
        for step in range(con_step,MAX_STEP):
            if coord.should_stop():
                    break
                
            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:tra_images, y_:tra_labels})            
            if step % 100 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op,feed_dict={x:tra_images, y_:tra_labels})
                tra_summary_writer.add_summary(summary_str, step)
                
#            if step % 200 == 0 or (step + 1) == MAX_STEP:
#                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
#                val_loss, val_acc = sess.run([loss, accuracy],
#                                             feed_dict={x:val_images,y_:val_labels})
#                print('**  Step %d, val loss = %.2f, val accuracy = %.2f  **' %(step, val_loss, val_acc))

                    
            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()  