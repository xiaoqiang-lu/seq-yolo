import pickle
import shutil

from tqdm import tqdm
import cv2
import numpy as np
import os
from seq_nms import *

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def VN(mon):
    '''
    change this path to your own original test dataset
    '''
    path = '/path/Test/' + mon + '/frames'
    video_names = []
    for b in os.listdir(path):
        for s in os.listdir(path + '/' + b):
            video_name = b + '_' + s
            video_names.append(video_name)

    return video_names


def pkl_split(path, mon, save_path):
    save_p = save_path + '/' + mon
    create_path(save_p)
    preds = pickle.load(open(path, 'rb'))
    v_names = VN(mon)
    for video_name in tqdm(v_names):
        one_video = {}
        for k, v in preds.items():
            if k.startswith(video_name):
                one_video[k] = v
        if one_video:
            one_video = dict(sorted(one_video.items(), key=lambda x:x[0]))
            pickle.dump(one_video, open(save_p + '/' + video_name + '.pkl', 'wb'), protocol=4)


SRC_PATH = 'runs/test/Day/a_e5'
PATH = SRC_PATH + '/src'
create_path(PATH)
for file in os.listdir(SRC_PATH):
    if file.endswith('.pkl'):
        shutil.copy(SRC_PATH + '/' + file,
                    PATH + '/' + file)

split = True
sn = True
sn_mons = ['Apr', 'Aug', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Sep']
hb = True

if split:
    MONTHS = ['Apr', 'Aug', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Sep']
    path = PATH
    for mon in MONTHS:
        for file in os.listdir(path):
            if file.startswith(mon):
                pkl_split(path + '/' + file, mon,
                          path + '_split')

if sn:
    ### As I am not familiar with mulit-process, you can change it to speed fast
    for MONTH in sn_mons:
        path = PATH + '_split/' + MONTH
        save_path = path.replace('src_split', 'src_split_seq_' + str(CONF_THRESH) + '_' + str(NMS_THRESH) + '_' + str(IOU_THRESH) + '_' + str(IOU_THRESH_DELETE))
        create_path(save_path)
        for file in tqdm(os.listdir(path)):
            dets = [[] for i in CLASSES[1:]]
            preds = pickle.load(open(path + '/' + file, 'rb'))
            for cls_ind, cls in enumerate(CLASSES[1:]):
                for k, v in preds.items():
                    single_img_num_boxes = len(v['boxes'])
                    cls_boxes = np.zeros((single_img_num_boxes, 4), dtype=np.float64)
                    cls_scores = np.zeros((single_img_num_boxes, 1), dtype=np.float64)
                    for box_ind, box in enumerate(v['boxes']):
                        cls_boxes[box_ind][0] = box[1]
                        cls_boxes[box_ind][1] = box[0]
                        cls_boxes[box_ind][2] = box[3]
                        cls_boxes[box_ind][3] = box[2]
                        # cls_boxes[box_ind][:] = box[:]

                        if str(v['labels'][box_ind] + 1) == cls:
                            cls_scores[box_ind][0] = v['scores'][box_ind]
                        else:
                            cls_scores[box_ind][0] = 0.00001
                    cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float64)
                    dets[cls_ind].append(cls_dets)


            boxes, classes, scores = dinms(dets, True)

            new_dict = {}
            for ind, (k, v) in enumerate(preds.items()):
                b = [list(j) for j in boxes[ind]]
                c = [i - 1 for i in classes[ind]]
                new_dict[k] = {'boxes': b,
                               'labels': c}

            pickle.dump(new_dict, open(save_path + '/' + file, 'wb'), protocol=4)

if hb:
    for m in ['Apr', 'Aug', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Sep']:
        path = PATH + '_split_seq_' + str(CONF_THRESH) + '_' + str(NMS_THRESH) + '_' + str(IOU_THRESH) + '_' + str(IOU_THRESH_DELETE) + '/' +  m
        big_dict = {}
        if os.path.exists(path):
            for file in tqdm(os.listdir(path)):
                preds = pickle.load(open(path + '/' + file, 'rb'))
                for k, v in preds.items():
                    if not v['boxes']:
                        v['boxes'] = [[0, 0, 0, 0]]
                        v['labels'] = [0]
                big_dict.update(preds)
            hb_save_path = PATH + '_split_seq_' + str(CONF_THRESH) + '_' + str(NMS_THRESH) + '_' + str(IOU_THRESH) + '_' + str(IOU_THRESH_DELETE) + '_hb/src'
            create_path(hb_save_path)
            pickle.dump(big_dict, open(hb_save_path + '/' + m + '.pkl', 'wb'), protocol=4)