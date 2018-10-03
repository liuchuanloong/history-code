# -*- coding: utf-8 -*-
# import sys
# # import importlib as imp
# # imp.reload(sys)
# # sys.setdefaultencoding('utf-8')
# import pandas as pd
import pickle
import json
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import numpy as np
import string

class image:
    def __init__(self,image_dir=''):
        self.image_dir = image_dir
        self.temp = {}
        self.coordinate={}
        self.filename = 'annotation_clean_train.txt'
        self.pos = []
        self.Efield = []
        self.count = 0
        self.file = os.path.join(os.getcwd(),'landmark_images')
        self.pfile = os.path.join(os.getcwd(),'pfile_train')

    def IsValidImage(self,pathfile):
        bValid = True
        try:
            Image.open(pathfile).verify()
        except:
            bValid = False
        return bValid

    def IS_JPEG(self,pathfile):
        if not os.path.exists(pathfile):
            print('ValidWarning:IsValidImage {} not exist'.format(pathfile))
        else:
            try:
                i = Image.open(pathfile)
                return i.format == 'JPEG'
            except IOError:
                print('ValidFata:{} not validimage!!!'.format(pathfile))
                return False
    def IS_VALID_JPEG(self,jpg_file):
        """判断JPG文件下载是否完整
        """
        if jpg_file.split('.')[-1].lower() == 'jpg':
            with open(jpg_file, 'rb') as f:
                f.seek(-2, 2)
                return f.read() == '\xff\xd9'  # 判定jpg是否包含结束字段

    def show(self,image):
        json_dir = os.path.join(os.path.join(IMAGE.pfile, class_tmp),
                               '{0}.json'.format(count))
        try:
            with open(json_dir, "rb+") as json_file:
                img_dict = json.load(json_file)
                img_list = img_dict['image']
                # img = np.asarray(img_list,np.uint8)
                img = cv2.cvtColor(np.asarray(img_list,np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imshow('re.jpeg',img)
                cv2.waitKey()
                cv2.imwrite("temp.jpeg",img)

        except EOFError:
            return {}

        # im = Image.fromarray(A)
        # im.save("your_file.jpeg")
        # try:
        # im01 = Image.fromarray(img)
        # # w, h = im01.size
        # draw = ImageDraw.Draw(im01)
        # draw.line([(IMAGE.temp['x1'], IMAGE.temp['y1']), (IMAGE.temp['x1'], IMAGE.temp['y2']),
        #            (IMAGE.temp['x2'], IMAGE.temp['y2']), (IMAGE.temp['x2'], IMAGE.temp['y1']),
        #            (IMAGE.temp['x1'], IMAGE.temp['y1'])],fill=255, width=5)
        # im01.show()
        #
        # except:
        #     print('Warning: Could not show image {}'.format(image))


    def dump(self,image):
        # try:
        # if IMAGE.IS_JPEG(image):
        img = np.asarray(Image.open(image).convert('RGB'),np.uint8)
        img_list = img.tolist()
        IMAGE.temp['image'] = img_list
        json_data = json.dumps(IMAGE.temp)

        if not os.path.exists(os.path.join(IMAGE.pfile, class_tmp)):
            os.makedirs(os.path.join(IMAGE.pfile, class_tmp))
        json_dir = os.path.join(os.path.join(IMAGE.pfile, class_tmp),
                               '{0}.json'.format(count))
        if not os.path.exists(json_dir):
            with open(json_dir, 'w+') as json_file:
                json_file.write(json_data)
        else:
            print('info:json file has already done!')
        # else:
        #     print('Warning: Could not dump image {}'.format(image))
        # except:
        #     print('Warning: Could not dump image {}'.format(image))


    def load(self):
        json_dir = os.path.join(os.path.join(IMAGE.pfile, class_tmp),
                               '{0}.json'.format(count))
        try:
            with open(json_dir, "r+") as json_file:
                img_dict = json.load(json_file)
                # 获取字典中内容，转为list
                return img_dict
        except EOFError:
            return {}

IMAGE = image()
with open(IMAGE.filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()
        IMAGE.temp = {}
        # 整行读取数据
        if not lines:
            break
        url_tmp, class_tmp, x1, x2, y1, y2 = [temp for temp in lines.split(' ')]

        IMAGE.pos.append(url_tmp)  # 添加新读取的数据
        IMAGE.Efield.append(class_tmp)
        count = len(IMAGE.Efield)

        image_dir = os.path.join(os.path.join(IMAGE.file, class_tmp),
                                 '{0}.jpg'.format(count))
        print(image_dir)

        # if not os.path.exists(image_dir):
        #     print('Image {} not exists.'.format(image_dir))
        #     continue

        IMAGE.image_dir = image_dir
        IMAGE.temp['url'] = url_tmp
        IMAGE.temp['id'] = class_tmp
        IMAGE.temp['x1'] = int(x1)
        IMAGE.temp['x2'] = int(x2)
        IMAGE.temp['y1'] = int(y1)
        IMAGE.temp['y2'] = int(y2)


        IMAGE.show(IMAGE.image_dir)
        # IMAGE.dump(IMAGE.image_dir)
        # img = IMAGE.load()
        # d = os.path.join(os.path.join(IMAGE.file, '580'),
        #                          '3089.jpg')
        # print(IMAGE.IS_VALID_JPEG(d))









