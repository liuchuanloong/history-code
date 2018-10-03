# -*- coding: utf-8 -*-
import os
import json
from PIL import Image, ImageDraw
import numpy as np
import cv2

class Dataset:
    def __init__(self):
        self.json_path =  os.path.join(os.getcwd(),'../annotations_landmarks/pfile_train')
        self.json_quene = []
        self.temp = {}
        self.train_image = []

    def load(self,Json_dir):
        try:
            with open(Json_dir, "rb+") as json_file:
                img_dict = json.load(json_file)
                # 获取字典中内容，转为list
                return img_dict
        except EOFError:
            return {}
    def preImageHelper(self,Json_dir):
        image_dict = Dataset.load(Json_dir)
        image_list = image_dict['image']
        img = cv2.cvtColor(np.asarray(image_list, np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imshow('re.jpeg', img)
        # cv2.waitKey()
        # cv2.imwrite("temp.jpeg", img)
        classes = image_dict['id']
        x1 = image_dict['x1']
        x2 = image_dict['x2']
        y1 = image_dict['y1']
        y2 = image_dict['y2']




Dataset = Dataset()
#获得root下的所有Json文件,读入图片数据
for root, sub_folders, files in os.walk(Dataset.json_path, topdown=True):
        for name in files:
            json_path = os.path.join(root, name)
            Dataset.json_quene.append(json_path)




