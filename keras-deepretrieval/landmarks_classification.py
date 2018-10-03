import os
import numpy as np
import tensorflow as tf
import cv2
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,Input
from keras.optimizers import SGD,Adam
from keras.utils import np_utils,plot_model
from sklearn.utils import class_weight as cw
import matplotlib.pyplot as plt
import time

class config:
    def __init__(self):

        self.classes = 6
        self.n_epochs = 500
        self.batch_size = 16
        self.train_annotationfile = os.path.join(
            os.getcwd(),'../../annotations_landmarks/landmarkdataset/landmark_xy.txt')
        self.train_root = '/home/liuchuanloong/amy/deep_retrieval/annotations_landmarks/landmarkdataset'
        self.vali = False
        self.val_root = ''
        self.val_annotationfile = ''

class data:
    def __init__(self):
        self.list_image = []
        self.list_label = []
        self.class_weight = {}

    def sampler(self,y_train):

        class_weight = cw.compute_class_weight('balanced',
                                                         np.unique(y_train),
                                                         y_train)
        return class_weight

    def predata(self,root,annotationfile,train = True):
        with open(annotationfile, 'r') as file_to_read:
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
            if train:
                self.weight = self.sampler(self.list_label)
                for key,value in enumerate(self.weight):
                    self.class_weight[key] = value
                return(temp,self.class_weight)
            else:
                return temp

## generate  image_batch label_batch
class batch_generator:
    def __init__(self,batch_size):
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
        for n in range(self.batch_size):
            image,label = temp[self.index]
            x, y = self.oneImageLable(image,label)
            self.X.append(x)
            self.Y.append(y)
            self.index += 1
            self.index = 0 if self.index + self.batch_size > len(temp[:,0]) else self.index
        return np.array(self.X), np.array(self.Y)

generated_images = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=True,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

C = config()
Data = data()
Gen = batch_generator(C.batch_size)

temp_train,class_weight = Data.predata(C.train_root,C.train_annotationfile)
if C.vali:
    temp_val = Data.predata(C.val_root,C.val_annotationfile,train=False)

############define a classification networks##############
input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

model = VGG16(input_tensor=img_input,weights='imagenet', include_top=False)
for i, layer in enumerate(model.layers):
   print(i, layer.name)
x = model.output
# x = K.batch_flatten(x)
x = GlobalAveragePooling2D()(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
predictions = Dense(C.classes, activation='softmax', name='predictions')(x)
new_model = Model(inputs=img_input,outputs=predictions)
############################################################

new_model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-3, momentum=0.9,decay=5*10e-5),
              metrics=['accuracy'])
log_path = './log'
callback = TensorBoard(log_path)
callback.set_model(new_model)
train_names = ['train_loss','train_acc']
losslist =[]

new_model.summary()
plot_model(new_model,'./tmp/new_model.png')
# log_filepath = './tmp/log'
# filepath="./tmp/log/OX_weights-{epoch:02d}-{val_acc:.2f}.hdf5"
# TB = TensorBoard(log_dir=log_filepath, write_images=0, histogram_freq=1)
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

try:
    new_model.load_weights('./tmp/VGG16_fine_tune_ox_Fri May  4 16:45:02 2018.h5')
    print('Success load weighs!')
except:
    print('Could not load model weights!')

for e in range(C.n_epochs):
    print('Training... \nEpoch', e)

    for n in range(int(temp_train.shape[0]/C.batch_size)):
        X_batch, Y_batch = Gen.generate_one_from_list(temp_train)
        generated_images.fit(X_batch)
        gen = generated_images.flow(X_batch, Y_batch, batch_size=C.batch_size,shuffle=False)
        X_batch, Y_batch = next(gen)
        # img = np.reshape(X_batch,(224,224,3))
        # img = np.asarray(img, np.uint8)
        # print(Y_batch)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('x.jpeg',img)
        # cv2.waitKey()

        Y_batch = np_utils.to_categorical(Y_batch, C.classes)
        loss = new_model.train_on_batch(X_batch, Y_batch,class_weight=class_weight)
        losslist.append(loss[0])
        print('Epoch:{} batch:{}/{} train_loss:{} trian_acc:{}'
            .format(e,n,int(temp_train.shape[0]/C.batch_size),loss[0],loss[1]))
        write_log(callback, train_names, loss, len(losslist))
    new_model.save('./tmp/VGG16_fine_tune_ox_{}.h5'.format(time.ctime()))
        ##################
        # plt.ion()
        # x = range(len(loss))
        # y = loss
        # plt.plot(x, y)
        # plt.pause(0.1)
        # with open('loss_log{}.txt'.format(time.ctime()), 'a') as f:
        # saveloss = '{}{}'.format(loss[0],'\n')
            # f.writelines(saveloss)

    # validation
    if C.vali:
        X_val_batch, Y_val_batch = Gen.generate_one_from_list(temp_val)
        generated_images.fit(X_val_batch)
        gen = generated_images.flow(X_val_batch, Y_val_batch, batch_size=C.batch_size, shuffle=False)
        X_val_batch, Y_val_batch = next(gen)

        Y_batch = np_utils.to_categorical(Y_val_batch, C.classes)
        val_loss = new_model.test_on_batch(X_val_batch, Y_val_batch)
        print('Epoch:{} val_loss:{}'.format(e, val_loss[0]))
new_model.save('./tmp/VGG16_fine_tune_ox_{}.h5'.format(time.ctime()))

