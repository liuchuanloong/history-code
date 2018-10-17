import os
cwd = os.getcwd()
PATH = '/media/loong/TOSHIBALOONG/TFKAGGLE/kaggle/cvtdata/train_labels/912b573a9e.png'
import cv2
img = cv2.imread(PATH)
print(img)