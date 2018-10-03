# -*- coding: utf-8 -*-
# @Author:Liu Chuanlong
import os
from urllib import request
from retry import retry
import shutil

class image:
    def __init__(self):
        self.filename = 'annotation_full_train.txt'
        self.success = []
        self.false = []
        self.count = 0
        self.count1 = 0
        self.temp = {}
        self.tmp_file = 'temp_full_train'
        self.landmarks = 'landmarks_full_train'
        self.txt = 'landmarks_full_train.txt'

def IS_VALID_JPEG(jpg_file):
    if not os.path.exists(jpg_file):
        print('ValidWarning:IsValidImage {} not exist'.format(jpg_file))
    else:
        try:
            if jpg_file.split('.')[-1].lower() == 'jpeg':
                with open(jpg_file, 'rb') as f:
                    f.seek(-2, 2)
                    return(f.read() == b'\xff\xd9')
        except:
            print('ValidFata:{} not validimage!!!'.format(jpg_file))

@retry(tries=3,delay=5)
def download(url_tmp,Image_dir,count):
    print('DownloadStart: download image {}'.format(Image_dir))
    f = request.urlopen(url_tmp, timeout=6)
    da = f.read()
    with open('./{}/{}.jpeg'.format(IMAGE.tmp_file,count), 'wb+') as code:
        code.write(da)
    print('DownloafDone:success download!!!')

IMAGE = image()
with open(IMAGE.filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline().strip()
        # 整行读取数据
        if not lines:
            break
        url_tmp, class_tmp = [temp for temp in lines.split(' ')]
        IMAGE.success.append(url_tmp)
        IMAGE.count = len(IMAGE.success) - 1
        IMAGE.temp['url'] = url_tmp
        IMAGE.temp['id'] = int(class_tmp)

        image_path = os.path.join(os.path.join(os.getcwd(),
                                              os.path.join(IMAGE.landmarks,class_tmp)),'{0}.jpg'.format(IMAGE.count))
        if not os.path.exists(os.path.join(os.getcwd(),
                                           os.path.join(IMAGE.landmarks,class_tmp))):
            os.makedirs(os.path.join(IMAGE.landmarks,class_tmp))
        if os.path.exists(image_path):
            print('Image {} already exists. Skipping download.'.format(image_path))
            continue
        try:
            download(IMAGE.temp['url'],image_path,IMAGE.count)
        except:
            IMAGE.false.append(IMAGE.temp['url'])
            IMAGE.count1 = len(IMAGE.false)
            print('FATA: Could not download image {} {} from {}'.format(IMAGE.count1, image_path, url_tmp))
        if IS_VALID_JPEG('./{}/{}.jpeg'.format(IMAGE.tmp_file,IMAGE.count)):
            try:
                shutil.copyfile('./{}/{}.jpeg'.format(IMAGE.tmp_file,IMAGE.count),image_path)
                counterpart_path = os.path.join(
                    os.path.join(IMAGE.landmarks, str(IMAGE.temp['id'])),
                    '{}.jpeg'.format(IMAGE.count))
                image_line = '{},{}{}'.format(counterpart_path.strip(),IMAGE.temp['id'],'\n')
                with open(IMAGE.txt, 'a') as file:
                    file.writelines(image_line)
                print('Success:save image as {}'.format(counterpart_path))
            except:
                counterpart_path = os.path.join(
                    os.path.join(IMAGE.landmarks, str(IMAGE.temp['id'])),
                    '{}.jpeg'.format(IMAGE.count))
                wrong_line = '{}{}'.format(counterpart_path, '\n')
                with open('WrongImages.txt', 'a') as filewrong:
                    filewrong.writelines(wrong_line)
                print('Warning:could not save image {}'.format(counterpart_path))
        else:
            print('Warning: Not valid image,continue!!!')