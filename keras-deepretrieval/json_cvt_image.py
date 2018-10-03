# -*- coding: utf-8 -*-
# Author: liuchuanloong.name
import json
from PIL import Image, ImageDraw
import os
import numpy as np

class dataset:
    def __init__(self):
        self.json_quene = []
        self.image = {}
        self.count = 0

    def load(self, Json_dir):
        try:
            with open(Json_dir, "rb+") as json_file:
                img_dict = json.load(json_file)
                # 获取字典中内容，转为list
                return img_dict
        except EOFError:
            return {}
Dataset = dataset()

root_file = os.path.join(os.getcwd(),'pfile_val')
dataset_path = os.path.join(os.getcwd(),'landmarks_images_val')
for root, sub_folders, files in os.walk(root_file, topdown=True):
        for name in files:
            json_path = os.path.join(root, name)
            Dataset.json_quene.append(json_path)
            Dataset.count = len(Dataset.json_quene)
            img_dict = Dataset.load(json_path)
            Dataset.image['classes'] = img_dict['id']
            Dataset.image['x1'] = img_dict['x1']
            Dataset.image['x2'] = img_dict['x2']
            Dataset.image['y1'] = img_dict['y1']
            Dataset.image['y2'] = img_dict['y2']
            Dataset.image['w'] = Dataset.image['x2'] - Dataset.image['x1']
            Dataset.image['h'] = Dataset.image['y2'] - Dataset.image['y1']
            Dataset.image['image'] = img_dict['image']

            image_path = os.path.join(dataset_path,str(Dataset.image['classes']))

            if not os.path.exists(image_path):
                os.makedirs(image_path)
            im = Image.fromarray(np.asarray(Dataset.image['image'],np.uint8))
            try:
                im.save(os.path.join(image_path,"{}.jpeg").format(Dataset.count))
                counterpart_path = os.path.join(
                    os.path.join('landmarks_images_val', str(Dataset.image['classes'])),
                    '{}.jpeg'.format(Dataset.count))
                image_line_xywh = '{} {} {} {} {} {} {}'.format(counterpart_path.strip(),
                                                                Dataset.image['x1'],
                                                                Dataset.image['y1'],
                                                                Dataset.image['w'],
                                                                Dataset.image['h'],
                                                                Dataset.image['classes'],
                                                                '\n')
                image_line_xy = '{},{},{},{},{},{}{}'.format(counterpart_path.strip(),
                                                             Dataset.image['x1'],
                                                             Dataset.image['y1'],
                                                             Dataset.image['x2'],
                                                             Dataset.image['y2'],
                                                             Dataset.image['classes'],
                                                             '\n')
                with open('annotations_landmarks_val_xywh.txt', 'a') as f1:
                    f1.writelines(image_line_xywh)
                with open('annotations_landmarks_val_xy.txt', 'a') as f2:
                    f2.writelines(image_line_xy)
                print('Success:save image as {}'.format(counterpart_path))
            except:
                counterpart_path = os.path.join(
                    os.path.join('landmarks_images_val', str(Dataset.image['classes'])),
                    '{}.jpeg'.format(Dataset.count))
                wrong_line = '{}{}'.format(counterpart_path,'\n')
                with open('WrongImages.txt', 'a') as f3:
                    f3.writelines(wrong_line)
                print('Warning:could not save image {}'.format(counterpart_path))





# ##################part two#############################
# class FILE:
#     def __init__(self):
#         self.coordinate = {}
# File  = FILE()
#
# with open('annotations_landmarks_xywh.txt', 'r') as file_to_read:
#     while True:
#         lines = file_to_read.readline().strip()
#         # 整行读取数据
#         if not lines:
#             break
#         path_temp, x, y, w, h,class_temp = [temp for temp in lines.split(' ')]
#         File.coordinate['x1'] = x
#         File.coordinate['y1'] = y
#         File.coordinate['x2'] = x + w
#         File.coordinate['y2'] = x + h
#         File.coordinate['id'] = new_path_temp
#         File.coordinate['class_temp'] = class_temp
#
#         image_line_xy = '{},{},{},{},{},{}{}'.format(File.coordinate['id'],
#                                                    File.coordinate['x1'],
#                                                    File.coordinate['y1'],
#                                                    File.coordinate['x2'],
#                                                    File.coordinate['y2'],
#                                                    File.coordinate['class_temp'],'\n')
#
#         with open('annotations_landmarks_xy.txt', 'a') as f:
#             f.write(image_line)



