# -*- coding: utf-8 -*-
import os
import urllib2
import pickle
import json
from PIL import Image, ImageDraw
import numpy as np
import string

class image:
    def __init__(self):
        self.filename = 'annotation_clean_val.txt'
        # txt文件和当前脚本在同一目录下，所以不用写具体路径
        self.success = []
        self.false = []
        self.count = 0
        self.count1 = 0

        self.temp = {}
        self.pfile = os.path.join(os.getcwd(),'pfile_val')

    def load(self,Json_dir):

        try:
            with open(Json_dir, "rb+") as json_file:
                img_dict = json.load(json_file)
                # 获取字典中内容，转为list
                return img_dict
        except EOFError:
            return {}

    def show(self,Json_dir):
        try:
            with open(Json_dir, "rb+".encode(encoding='utf-8')) as json_file:
                img_dict = json.load(json_file)
                img_list = img_dict['image']
                id = img_dict['id']
                img = np.asarray(img_list,np.uint8)
                IMAGE.temp['x1'] = img_dict['x1']
                IMAGE.temp['x2'] = img_dict['x2']
                IMAGE.temp['y1'] = img_dict['y1']
                IMAGE.temp['y2'] = img_dict['y2']
        except:
            print('SHOW@1:could not read json image path {}'.format(Json_dir))
        try:
            im01 = Image.fromarray(img)
            draw = ImageDraw.Draw(im01)
            draw.line([(IMAGE.temp['x1'], IMAGE.temp['y1']), (IMAGE.temp['x1'], IMAGE.temp['y2']),
                       (IMAGE.temp['x2'], IMAGE.temp['y2']), (IMAGE.temp['x2'], IMAGE.temp['y1']), (IMAGE.temp['x1'], IMAGE.temp['y1'])],
                      fill=(255, 0, 0), width=5)
            im01.show()
        except:
            print('SHOW@2: Could not show image {}'.format(Json_dir))

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
        if not os.path.exists(jpg_file):
            print('ValidWarning:IsValidImage {} not exist'.format(jpg_file))
        else:
            try:
                if jpg_file.split('.')[-1].lower() == 'jpg':
                    with open(jpg_file, 'rb') as f:
                        f.seek(-2, 2)
                        return f.read() == '\xff\xd9'
            except:
                print('ValidFata:{} not validimage!!!'.format(jpg_file))

    def IsValidImage(self,pathfile):
        if not os.path.exists(pathfile):
            print('ValidWarning:IsValidImage {} not exist'.format(pathfile))
        else:
            bValid = True
            try:
                Image.open(pathfile).verify()
            except:
                bValid = False
                print('ValidFata:{} not validimage!!!'.format(pathfile))
            return bValid
    def dump(self,img_dir,json_dir):
        if IMAGE.IS_VALID_JPEG(img_dir):
            img = np.array(Image.open(img_dir))
            img_list = img.tolist()
            IMAGE.temp['image'] = img_list
            json_data = json.dumps(IMAGE.temp)

            if not os.path.exists(json_dir):
                with open(json_dir, 'wb+') as json_file:
                    json_file.write(json_data)
                print('{}!!!!DONE!!!!'.format(json_dir))
            else:
                print('JsonDone:json file {} has already done!'.format(json_dir))
        else:
            print('Warning: Could not dump image {}'.format(json_dir))


IMAGE = image()

with open(IMAGE.filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline().strip()
        # 整行读取数据
        if not lines:
            break
        url_tmp, class_tmp, x1, x2, y1, y2 = [temp for temp in lines.split(' ')] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        IMAGE.success.append(url_tmp)  # 添加新读取的数据
        IMAGE.count = len(IMAGE.success)

        IMAGE.temp['url'] = url_tmp
        IMAGE.temp['id'] = int(class_tmp)
        IMAGE.temp['x1'] = int(x1)
        IMAGE.temp['x2'] = int(x2)
        IMAGE.temp['y1'] = int(y1)
        IMAGE.temp['y2'] = int(y2)

        Image_dir = os.path.join(os.path.join(os.getcwd(),
                                              os.path.join('landmark_temp_val',class_tmp)),'{0}.jpg'.format(IMAGE.count))
        if not os.path.exists(os.path.join(os.getcwd(),
                                           os.path.join('landmark_temp_val',class_tmp))):
            os.makedirs(os.path.join('landmark_temp_val',class_tmp))
        if not os.path.exists(os.path.join(IMAGE.pfile, class_tmp)):
            os.makedirs(os.path.join(IMAGE.pfile, class_tmp))
        json_dir = os.path.join(os.path.join(IMAGE.pfile, class_tmp),
                                '{0}.json'.format(IMAGE.count))
        if os.path.exists(Image_dir):
            print('Image {} already exists. Skipping download.'.format(Image_dir))
            IMAGE.dump(Image_dir,json_dir)
            # IMAGE.show(json_dir)
            continue
        try:
            print('DownloadStart: download image {}'.format(Image_dir))
            f = urllib2.urlopen(url_tmp, timeout=5)
            da = f.read()
            with open(Image_dir, 'wb+') as code:
                code.write(da)
            print('DownloafDone:image{} success download!!!'.format(Image_dir))
        except:
            IMAGE.false.append(url_tmp)
            IMAGE.count1 = len(IMAGE.false)
            print('FATA: Could not download image {} {} from {}'.format(IMAGE.count1,Image_dir, url_tmp))

        IMAGE.dump(Image_dir,json_dir)
        # IMAGE.show(json_dir)
