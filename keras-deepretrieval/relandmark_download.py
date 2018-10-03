# -*- coding: utf-8 -*-
import os
import urllib2
import pickle
import json
from PIL import Image, ImageDraw
import numpy as np
import string
import hashlib
from retry import retry
import shutil
import time

class image:
    def __init__(self):
        self.filename = 'annotation_clean_val.txt'
        self.md5path = 'annotation_clean_val.md5sum'
        self.success = []
        self.false = []
        self.count = 0
        self.count1 = 0

        self.temp = {}

    def GetFileMd5(self,filename):
        if not os.path.exists(filename):
            print('tempfile is not exists,get MD5 faile!!!')
            return
        print('Success Get Image MD5')
        myhash = hashlib.md5()
        f = file(filename, 'rb')
        while True:
            b = f.read(8096)
            if not b:
                break
            myhash.update(b)
        f.close()
        print(myhash.hexdigest())
        return myhash.hexdigest()

IMAGE = image()
md5list = []


def IS_VALID_JPEG(jpg_file):
    """判断JPG文件下载是否完整
    """
    if not os.path.exists(jpg_file):
        print('ValidWarning:IsValidImage {} not exist'.format(jpg_file))
    else:
        try:
            if jpg_file.split('.')[-1].lower() == 'jpeg':
                with open(jpg_file, 'rb') as f:
                    f.seek(-2, 2)
                    return f.read() == '\xff\xd9'  # 判定jpg是否包含结束字段
        except:
            print('ValidFata:{} not validimage!!!'.format(jpg_file))
@retry(tries=3,delay=5)
def download(url_tmp,Image_dir,count):
    print('DownloadStart: download image {}'.format(Image_dir))
    f = urllib2.urlopen(url_tmp, timeout=6)
    da = f.read()
    with open('./temp_val/{}.jpeg'.format(count), 'wb+') as code:
        code.write(da)
    print('DownloafDone:success download!!!')



with open(IMAGE.md5path, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline().strip()
        # 整行读取数据
        if not lines:
            break
        md5list.append(lines)
with open(IMAGE.filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline().strip()
        # 整行读取数据
        if not lines:
            break
        url_tmp, class_tmp, x1, x2, y1, y2 = [temp for temp in lines.split(' ')] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        IMAGE.success.append(url_tmp)  # 添加新读取的数据
        IMAGE.count = len(IMAGE.success) - 1

        IMAGE.temp['url'] = url_tmp
        IMAGE.temp['id'] = int(class_tmp)
        IMAGE.temp['x1'] = int(x1)
        IMAGE.temp['x2'] = int(x2)
        IMAGE.temp['y1'] = int(y1)
        IMAGE.temp['y2'] = int(y2)

        Image_dir = os.path.join(os.path.join(os.getcwd(),
                                              os.path.join('landmarks_val',class_tmp)),'{0}.jpg'.format(IMAGE.count))
        if not os.path.exists(os.path.join(os.getcwd(),
                                           os.path.join('landmarks_val',class_tmp))):
            os.makedirs(os.path.join('landmarks_val',class_tmp))
        if os.path.exists(Image_dir):
            print('Image {} already exists. Skipping download.'.format(Image_dir))
            continue
        try:
            download(IMAGE.temp['url'],Image_dir,IMAGE.count)
        except:
            IMAGE.false.append(IMAGE.temp['url'])
            IMAGE.count1 = len(IMAGE.false)
            print('FATA: Could not download image {} {} from {}'.format(IMAGE.count1, Image_dir, url_tmp))
        print "Start : %s" % time.ctime()
        time.sleep(1)
        print "End : %s" % time.ctime()
        if IS_VALID_JPEG('./temp_val/{}.jpeg'.format(IMAGE.count)):
            md5 = IMAGE.GetFileMd5('./temp_val/{}.jpeg'.format(IMAGE.count))
            print(md5list[IMAGE.count])
            if md5 == md5list[IMAGE.count]:
                try:
                    shutil.copyfile('./temp_val/{}.jpeg'.format(IMAGE.count),Image_dir)
                    counterpart_path = os.path.join(
                        os.path.join('landmarks_val', str(IMAGE.temp['id'])),
                        '{}.jpeg'.format(IMAGE.count))
                    image_line_xy = '{},{},{},{},{},{}{}'.format(counterpart_path.strip(),
                                                                 IMAGE.temp['x1'],
                                                                 IMAGE.temp['y1'],
                                                                 IMAGE.temp['x2'],
                                                                 IMAGE.temp['y2'],
                                                                 IMAGE.temp['id'],
                                                                 '\n')
                    with open('landmarks_val_xy.txt', 'a') as f2:
                        f2.writelines(image_line_xy)
                    print('Success:save image as {}'.format(counterpart_path))
                except:
                    counterpart_path = os.path.join(
                        os.path.join('landmarks_val', str(IMAGE.temp['id'])),
                        '{}.jpeg'.format(IMAGE.count))
                    wrong_line = '{}{}'.format(counterpart_path, '\n')
                    with open('WrongImages.txt', 'a') as f3:
                        f3.writelines(wrong_line)
                    print('Warning:could not save image {}'.format(counterpart_path))
            else:
                print('MD5 not Matching!!!!!')
        else:
            print('Warning: Is not valid image,continue!!!')
