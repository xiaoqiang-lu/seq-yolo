import os
import random
import shutil
import cv2
from tqdm import tqdm
import numpy as np

names = ['human', 'bicycle', 'motorcycle', 'vehicle']

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert(txt_path, save_path):

    f = open(txt_path, 'r')
    wr = open(save_path, 'a')
    annos = f.readlines()
    for anno in annos:
        info = anno.strip().split(' ')
        x, y, x2, y2 = map(int, info[2:6])
        x_c_n = format(((x + x2) / 2) / 384, '.6f')
        y_c_n = format(((y + y2) / 2) / 288, '.6f')
        w_n = format((x2 - x) / 384, '.6f')
        h_n = format((y2 - y) / 288, '.6f')
        category = names.index(info[1])
        wr.write(str(category) + ' ' + str(x_c_n) + ' ' + str(y_c_n) + ' ' + str(w_n) + ' ' + str(h_n) + '\n')


F_PATH = '/path'
SPLIT = 'Day'

PATH = F_PATH + '/Train/' + SPLIT
img_save_path = F_PATH + '/together/Train/' + SPLIT + '/images'
label_save_path = F_PATH + '/together/Train/' + SPLIT + '/labels'
create_path(img_save_path)
create_path(label_save_path)

for date in tqdm(os.listdir(PATH + '/annotations')):
    for clip in os.listdir(PATH + '/annotations/' + date):
        for file in os.listdir(PATH + '/annotations/' + date + '/' + clip):
            if os.path.getsize(PATH + '/annotations/' + date + '/' + clip + '/' + file):
                shutil.copy(PATH + '/frames/' + date + '/' + clip + '/' + file.replace('annotations', 'image').replace('.txt', '.jpg'),
                            img_save_path + '/' + date + '_' + clip + '_' + file.split('_')[1].replace('.txt', '.jpg'))

                convert(PATH + '/annotations/' + date + '/' + clip + '/' + file,
                        label_save_path + '/' + date + '_' + clip + '_' + file.split('_')[1])