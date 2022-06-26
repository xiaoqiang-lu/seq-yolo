import json
import os
import pickle
import shutil

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


# data = pickle.load(open('/root/lxq/ECCV2022/test_pkl_v4/Day/all_every10frames/p6_ep100_bs32_img1280_scratch/Day_all_every10frames_v4p6_1280_59_69_79_89_99_0.6_0.1_Valid/submit/0.1/predictions.pkl', 'rb'))
# # json.dump(data, open('./val_submit_example.json', 'w'))
# # print(len(data))
# for k, v in data['Sep'].items():
#     print(v)

# f = open('./val_submit_example.json')
# infos = json.load(f)
# apr_infos = infos['Sep']
# print(apr_infos)
# for k, v in apr_infos.items():
#     print(k)


# src_path = '/root/lxqbigdata/CVPR2022/UG2+/Track1/wbf/1/jsons'
# json_names = os.listdir(src_path)
# json_names.sort()
# print(json_names)

# path = '/root/lxq/ug2+/dataset/v4_data/clean_haze_allflip/train/images'
# for file in tqdm(os.listdir(path)):
#     img = cv2.imread(path + '/' + file)
#     img_flip = cv2.flip(img, 1)
#     cv2.imwrite('/root/lxq/ug2+/dataset/v4_data/clean_haze_allflip/train/images/' + file.replace('.jpg', '_flip.jpg'), img_flip)



def flip_result2normal(json_file):
    preds = json.load(open(json_file))
    for pred in tqdm(preds):
        img = cv2.imread('/root/lxq/ug2+/dataset/v4_data/clean_haze/test/images/' + pred['image_id'] + '.jpg')
        _, img_w, _ = img.shape
        [x, y, w, h] = pred['bbox']
        pred['bbox'] = [img_w - x - w, y, w, h]
    json.dump(preds, open(json_file.replace('.json', '_normal.json'), 'w'))


# flip_result2normal('/root/lxqbigdata/CVPR2022/UG2+/Track1/scaled_yolov4/runs/final_test/clean_haze/v4p6_ep50_bs16_imgsize1280_ddp_prep6_finetune_t+v/v4p6_flip666666_1664_last_045_0.4_0.001_aug.json')





# label_path = '/root/lxq/ug2+/dataset/v4_data/clean_haze_allflip/train/labels'
# for file in tqdm(os.listdir(label_path)):
#     f = open(label_path + '/' + file, 'r')
#     w = open(label_path + '/' + file.replace('.txt', '_flip.txt'), 'a')
#     annos = f.readlines()
#     for anno in annos:
#         info = anno.strip().split(' ')
#         flip_info = 1 - float(info[1])
#         w.write(info[0] + ' ' + str(flip_info) + ' ' + info[2] + ' ' + info[3] + ' ' + info[4] + '\n')


# path = '/root/lxqbigdata/CVPR2022/Fisheye/code/s_yv4/runs/test/p6_ep100_bs16_img1280_all_finetune/p6_1536_last_099_0.6_0.4/object_detection_v'
# for file in tqdm(os.listdir(path)):
#     img = np.array(Image.open(path + '/' + file))
#     cv2.imwrite('/root/lxqbigdata/CVPR2022/Fisheye/code/s_yv4/runs/test/p6_ep100_bs16_img1280_all_finetune/p6_1536_last_099_0.6_0.4/472/' +
#                 file.replace('.png', '.jpg'), img)


# path = '/root/lxq/ECCV2022/Chalearn/together/Valid/all/images'
# # for mon in os.listdir(path):
# #     for img in tqdm(os.listdir(path + '/' + mon + '/images')):
# #         os.rename(path + '/' + mon + '/images' + '/' + img,
# #                   path + '/' + mon + '/images' + '/' + img.replace('__', '_'))
# for img in tqdm(os.listdir(path)):
#     os.rename(path + '/' + img,
#               path + '/' + img.replace('__', '_'))


# f = open('/root/lxq/ECCV2022/Chalearn/together/Valid/all/test.txt', 'r')
# names = f.readlines()
# num = len(names)
# n = int(num / 4)
# for i in range(4):
#     wr = open('/root/lxq/ECCV2022/Chalearn/together/Valid/all/test_' + str(i) + '.txt', 'a')
#     if i < 3:
#         for name in names[i * n:(i + 1) * n]:
#             wr.write(name)
#     else:
#         for name in names[i * n:]:
#             wr.write(name)



# data = pickle.load(open('/root/lxq/ECCV2022/test_json/Day/split_8_2/p5_ep100_bs32_img896_all_finetune/Day_split82_p5_896_best_0.6_0.1_Valid.pkl', 'rb'))
# for d in tqdm(data):
#     preds = data[d]
#     pred_names, all_names = [], []
#     for pred in preds:
#         pred_names.append(pred)
#     for name in os.listdir('/root/lxq/ECCV2022/Chalearn/together/Valid/split/' + d + '/images'):
#         all_names.append(name.split('.')[0])
#     not_names = list(set(all_names) - set(pred_names))
#     for not_name in not_names:
#         preds[not_name] = {'boxes': [[0, 0, 0, 0]],
#                            'labels': [0]}
# wr = open('/root/lxq/ECCV2022/test_json/Day/split_8_2/p5_ep100_bs32_img896_all_finetune/Day_split82_p5_896_best_0.6_0.1_Valid_allname.pkl', 'wb')
# pickle.dump(data, wr, protocol=4)


# path = '/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/labels'
# a = []
# for file in tqdm(os.listdir(path)):
#     f = open(path + '/' + file, 'r')
#     infos = f.readlines()
#     for info in infos:
#         label = info.strip().split(' ')
#         if label[0] == '1' or label[0] == '2' or label[0] == '3':
#             a.append(file)
# print(len(set(a)))



# f = open('/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/all.txt', 'r')
# names = f.readlines()
# a = []
# for name in names:
#     img_name = name.strip().split('/')[-1]
#     r = open('/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/labels/' + img_name.replace('.jpg', '.txt'))
#     infos = r.readlines()
#     for info in infos:
#         label = info.strip().split(' ')
#         # if label[0] == '1' or label[0] == '2' or label[0] == '3':
#         if label[0] == '2':
#             a.append(img_name)
# motor_names = set(a)
# e5 = open('/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/all_every10frames.txt', 'r')
# e5_names = e5.readlines()
# motor_list = []
# for motor_name in motor_names:
#     motor_list.append('./images/' + motor_name + '\n')
# final = set(e5_names + motor_list)
# wr = open('/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/all_every10frames_wallmotor.txt', 'a')
# for n in final:
#     wr.write(n)



# f = open('/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/all_every5frames_wallmotor.txt', 'r')
# names = f.readlines()
# for name in tqdm(names):
#     img_name = name.strip().split('/')[-1]
#     shutil.copy('/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/images/' + img_name,
#                 '/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/day933/images/' + img_name)
#     shutil.copy('/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/labels/' + img_name.replace('.jpg', '.txt'),
#                 '/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/train_v4/day933/labels/' + img_name.replace('.jpg', '.txt'))


def VN(mon):
    path = '/root/lxq/ECCV2022/Chalearn/Test/' + mon + '/frames'
    video_names = []
    for b in os.listdir(path):
        for s in os.listdir(path + '/' + b):
            video_name = b + '_' + s
            video_names.append(video_name)

    return video_names


# MONTH = 'May'    # [Apr, Aug, Jan, Jul, Jun, Mar, May, Sep]
# for MONTH in ['Apr', 'Aug', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Sep']:
#     path = 'runs/final_test/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch/Day_a_e10_bic10_mot_1280_59-99-5_ms_51/src_split_seq/' + MONTH
#     big_dict = {}
#     for file in tqdm(os.listdir(path)):
#         preds = pickle.load(open(path + '/' + file, 'rb'))
#         for k, v in preds.items():
#             if not v['boxes']:
#                 v['boxes'] = [[0, 0, 0, 0]]
#                 v['labels'] = [0]
#         big_dict.update(preds)
#     pickle.dump(big_dict, open('runs/final_test/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch/Day_a_e10_bic10_mot_1280_59-99-5_ms_51/src_split_seq/src/' + MONTH + '.pkl', 'wb'),
#                 protocol=4)

# print(len(pickle.load(open('/root/lxqbigdata/ECCV2022/Chalearn/code/v4/runs/test/Day/all_every5frames/p7_ep100_bs16_img1536_scratch/Day_all_every5frames_v4p7_1536_29_39_49_59_69_79_89_99_0.6_0.1_Valid/src/Sep_1536_29_39_49_59_69_79_89_99_ms_0.6_0.1.pkl', 'rb'))))














f = open('/root/lxq/ECCV2022/Chalearn/together/Train/Month/train/split/train_v4/all.txt', 'r')
names = f.readlines()
a = []
for name in names:
    img_name = name.strip().split('/')[-1]
    r = open('/root/lxq/ECCV2022/Chalearn/together/Train/Month/train/split/train_v4/labels/' + img_name.replace('.jpg', '.txt'))
    infos = r.readlines()
    for info in infos:
        label = info.strip().split(' ')
        # if label[0] == '1' or label[0] == '2' or label[0] == '3':
        if label[0] == '2':
            a.append(img_name)
motor_names = set(a)
motor_list = []
for motor_name in motor_names:
    motor_list.append('./images/' + motor_name + '\n')

wr = open('/root/lxq/ECCV2022/Chalearn/together/Train/Month/train/split/train_v4/Motorcycle.txt', 'a')
for n in motor_list:
    wr.write(n)
