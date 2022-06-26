import math
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


def create_testset(path, save_path):
    for mon in os.listdir(path):
        save_dir = save_path + '/' + mon + '/images'
        create_path(save_dir)
        for b_video in tqdm(os.listdir(path + '/' + mon + '/frames')):
            for s_video in os.listdir(path + '/' + mon + '/frames' + '/' + b_video):
                for img in os.listdir(path + '/' + mon + '/frames' + '/' + b_video + '/' + s_video):
                    shutil.copy(path + '/' + mon + '/frames' + '/' + b_video + '/' + s_video + '/' + img,
                                save_dir + '/' + b_video + '_' + s_video + '_' + img.split('_')[1])

# create_testset('/root/lxq/ECCV2022/Chalearn/Test',
#                '/root/lxq/ECCV2022/Chalearn/together/Test/split')


def convert(txt_path, save_path):
    create_path(save_path)
    for file in tqdm(os.listdir(txt_path)):
        f = open(txt_path + '/' + file, 'r')
        wr = open(save_path + '/' + file, 'a')
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



# MONTH = 'May'    #[Apr, Aug, Jan, Jul, Jun, Mar, May, Sep]
# path = '/root/lxq/ECCV2022/Chalearn/together/Valid/labels/Valid/' + MONTH + '/annotations'
# save_path = '/root/lxq/ECCV2022/Chalearn/together/Valid/labels/tog/' + MONTH + '/labels_src'
# create_path(save_path)
#
# for date in os.listdir(path):
#     for clip in tqdm(os.listdir(path + '/' + date)):
#         for img in os.listdir(path + '/' + date + '/' + clip):
#             if img:
#                 name = img.split('annotations')[1]
#                 shutil.copy(path + '/' + date + '/' + clip + '/' + img,
#                             save_path + '/' + date + '_' + clip + name)
#
# convert(save_path, '/root/lxq/ECCV2022/Chalearn/together/Valid/' + MONTH + '/labels')


# name = 'Month'
# path = '/root/lxq/ECCV2022/Chalearn/together/' + name + '/train/src/labels'
# img_path = '/root/lxq/ECCV2022/Chalearn/together/' + name + '/train/src/images'
# labeled_img_save_path = '/root/lxq/ECCV2022/Chalearn/together/' + name + '/train/split/labeled/images'
# labeled_label_save_path = '/root/lxq/ECCV2022/Chalearn/together/' + name + '/train/split/labeled/labels'
# unlabeled_img_save_path = '/root/lxq/ECCV2022/Chalearn/together/' + name + '/train/split/unlabeled/images'
# unlabeled_label_save_path = '/root/lxq/ECCV2022/Chalearn/together/' + name + '/train/split/unlabeled/labels'
# create_path(labeled_img_save_path)
# create_path(labeled_label_save_path)
# create_path(unlabeled_img_save_path)
# create_path(unlabeled_label_save_path)
# for file in tqdm(os.listdir(path)):
#     if os.path.getsize(path + '/' + file):
#         shutil.copy(img_path + '/' + file.replace('.txt', '.jpg'),
#                     labeled_img_save_path + '/' + file.replace('.txt', '.jpg'))
#         shutil.copy(path + '/' + file,
#                     labeled_label_save_path + '/' + file)
#     else:
#         shutil.copy(img_path + '/' + file.replace('.txt', '.jpg'),
#                     unlabeled_img_save_path + '/' + file.replace('.txt', '.jpg'))
#         shutil.copy(path + '/' + file,
#                     unlabeled_label_save_path + '/' + file)


def txt_frames_sample(txt_file, sample_num):
    f = open(txt_file, 'r')
    wr = open(txt_file.replace('all.txt', 'all_e' + str(sample_num) + '.txt'), 'a')
    names = f.readlines()
    for i in range(len(names)):
        if i % sample_num == 0:
            wr.write(names[i])

def convert_label(txt_path, img_path, save_path):
    create_path(save_path)
    for file in tqdm(os.listdir(txt_path)):
        f = open(txt_path + '/' + file, 'r')
        wr = open(save_path + '/' + file, 'a')
        annos = f.readlines()
        w, h = 384, 288
        for anno in annos:
            info = anno.strip().split(' ')
            x, y, x2, y2 = map(int, info[2:6])
            x_c_n = format(((x + x2) / 2) / w, '.6f')
            y_c_n = format(((y + y2) / 2) / h, '.6f')
            w_n = format((x2 - x) / w, '.6f')
            h_n = format((y2 - y) / h, '.6f')
            category = names.index(info[1])
            wr.write(str(category) + ' ' + str(x_c_n) + ' ' + str(y_c_n) + ' ' + str(w_n) + ' ' + str(h_n) + '\n')

# convert_label('/root/lxq/ECCV2022/Chalearn/together/Train/Week/train/split/train_v4/labels_src',
#               '/root/lxq/ECCV2022/Chalearn/together/Train/Week/train/split/train_v4/images',
#               '/root/lxq/ECCV2022/Chalearn/together/Train/Week/train/split/train_v4/labels')

def writ_txt(path, save_path):
    wr = open(save_path, 'a')
    for file in sorted(os.listdir(path)):
        wr.write('./images/' + file + '\n')


def sample_txt(path, train_txt, val_txt):
    f = open(path, 'r')
    wrt = open(train_txt, 'a')
    wrv = open(val_txt, 'a')
    names = f.readlines()
    val_list = []
    for i in random.sample(range(0, 4385), int(0.2 * len(names))):
        val_list.append(names[i])
    train_list = list(set(names) - set(val_list))
    train_list, val_list = sorted(train_list), sorted(val_list)
    for info in train_list:
        wrt.write(info)
    for info in val_list:
        wrv.write(info)

# sample_txt('/root/lxq/ECCV2022/Chalearn/together/Day/train/split/labeled/train/all.txt',
#            '/root/lxq/ECCV2022/Chalearn/together/Day/train/split/labeled/train/train.txt',
#            '/root/lxq/ECCV2022/Chalearn/together/Day/train/split/labeled/train/val.txt')


# writ_txt('/root/lxq/ECCV2022/Chalearn/together/Test/split/Mar/images',
#          '/root/lxq/ECCV2022/Chalearn/together/Test/split/Mar/all.txt')

# f = open('/root/lxq/ECCV2022/Chalearn/together/Test/split/Mar/all.txt', 'r')
# ids = f.readlines()
# num = int(len(ids) / 4)
# for i in range(4):
#     wr = open('/root/lxq/ECCV2022/Chalearn/together/Test/split/Mar/' + str(i) + '.txt', 'a')
#     if i < 3:
#         for line in ids[i * num: (i + 1) * num]:
#             wr.write(line)
#     else:
#         for line in ids[i * num:]:
#             wr.write(line)



# path = '/root/lxq/ECCV2022/Chalearn/together/Valid/split'
# save_path = '/root/lxq/ECCV2022/Chalearn/together/Valid/all/images'
# n = 0
# for mon in tqdm(os.listdir(path)):
#     for img in os.listdir(path + '/' + mon + '/images'):
#         shutil.copy(path + '/' + mon + '/images' + '/' + img,
#                     save_path + '/' + mon + '_' + img)
#         n += 1
# print(n)


# img_path = '/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/labeled/train/images'
# label_path = '/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/labeled/train/labels_src'
# img_save_path = '/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/labeled/grid_img/val/images'
# label_save_path = '/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/labeled/grid_img/val/labels'
# create_path(img_save_path)
# create_path(label_save_path)
# f = open('/root/lxq/ECCV2022/Chalearn/together/Train/Day/train/split/labeled/train/val.txt', 'r')
# img_names = f.readlines()
# num = len(img_names)
# for i in tqdm(range(int(num / 9) + 1)):
#     b_img = np.zeros((864, 1152), dtype=int)
#     wr = open(label_save_path + '/' + str(i) + '.txt', 'a')
#     if i < 389:
#         for ind, name in enumerate(img_names[i * 9: (i + 1) * 9]):
#             img_dir = name.strip()
#             img_name = img_dir.split('/')[2]
#             img = cv2.imread(img_path + '/' + img_name, cv2.IMREAD_UNCHANGED)
#             h, w = math.floor(ind / 3), ind % 3
#             b_img[h * 288: (h + 1) * 288, w * 384: (w + 1) * 384] = img
#             cv2.imwrite(img_save_path + '/' + str(i) + '.jpg', b_img)
#
#             l_f = open(label_path + '/' + img_name.replace('.jpg', '.txt'), 'r')
#             labels = l_f.readlines()
#             for label in labels:
#                 info = label.strip().split(' ')
#                 x, y, x2, y2 = map(int, info[2:6])
#                 category = names.index(info[1])
#                 b_x, b_x2 = x + w * 384, x2 + w * 384
#                 b_y, b_y2 = y + h * 288, y2 + h * 288
#
#                 x_c_n = format(((b_x + b_x2) / 2) / 1152, '.6f')
#                 y_c_n = format(((b_y + b_y2) / 2) / 864, '.6f')
#                 w_n = format((b_x2 - b_x) / 1152, '.6f')
#                 h_n = format((b_y2 - b_y) / 864, '.6f')
#                 wr.write(str(category) + ' ' + str(x_c_n) + ' ' + str(y_c_n) + ' ' + str(w_n) + ' ' + str(h_n) + '\n')
#
#     else:
#         for ind, name in enumerate(img_names[num - 9:]):
#             img_dir = name.strip()
#             img_name = img_dir.split('/')[2]
#             img = cv2.imread(img_path + '/' + img_name, cv2.IMREAD_UNCHANGED)
#             h, w = math.floor(ind / 3), ind % 3
#             b_img[h * 288: (h + 1) * 288, w * 384: (w + 1) * 384] = img
#             cv2.imwrite(img_save_path + '/' + str(i) + '.jpg', b_img)
#
#             l_f = open(label_path + '/' + img_name.replace('.jpg', '.txt'), 'r')
#             labels = l_f.readlines()
#             for label in labels:
#                 info = label.strip().split(' ')
#                 x, y, x2, y2 = map(int, info[2:6])
#                 category = names.index(info[1])
#                 b_x, b_x2 = x + w * 384, x2 + w * 384
#                 b_y, b_y2 = y + h * 288, y2 + h * 288
#
#                 x_c_n = format(((b_x + b_x2) / 2) / 1152, '.6f')
#                 y_c_n = format(((b_y + b_y2) / 2) / 864, '.6f')
#                 w_n = format((b_x2 - b_x) / 1152, '.6f')
#                 h_n = format((b_y2 - b_y) / 864, '.6f')
#                 wr.write(str(category) + ' ' + str(x_c_n) + ' ' + str(y_c_n) + ' ' + str(w_n) + ' ' + str(h_n) + '\n')
import pickle


def VN(mon):
    path = '/root/lxq/ECCV2022/Chalearn/Test/' + mon + '/frames'
    video_names = []
    for b in os.listdir(path):
        for s in os.listdir(path + '/' + b):
            video_name = b + '_' + s
            video_names.append(video_name)

    return video_names


# path = '/root/lxqbigdata/ECCV2022/Chalearn/code/v4/runs/test/Day/all_every5frames/p7_ep100_bs16_img1536_scratch/Day_all_every5frames_v4p7_1536_29_39_49_59_69_79_89_99_0.6_0.1_Valid/src/Sep_1536_29_39_49_59_69_79_89_99_ms_0.6_0.1.pkl'

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


# MONTH = ['Apr', 'Aug', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Sep']
# path = 'runs/final_test/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch/Day_a_e10_bic10_mot_1280_59-99-5_ms_51/src'
# for mon in MONTH:
#     for file in os.listdir(path):
#         if file.startswith(mon):
#             pkl_split(path + '/' + file, mon,
#                       'runs/final_test/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch/Day_a_e10_bic10_mot_1280_59-99-5_ms_51/src_split')



# txt_frames_sample('/root/lxq/ECCV2022/Chalearn/together/Train/Month/train/split/train_v4/all.txt', 15)


# SPLIT = 'Month'
# f_e10 = open('/root/lxq/ECCV2022/Chalearn/together/Train/' + SPLIT + '/train/split/train_v4/all_e10.txt', 'r')
# f_Bic = open('/root/lxq/ECCV2022/Chalearn/together/Train/' + SPLIT + '/train/split/train_v4/Bicycle.txt', 'r')
# f_Mot = open('/root/lxq/ECCV2022/Chalearn/together/Train/' + SPLIT + '/train/split/train_v4/Motorcycle.txt', 'r')
# e10_names = f_e10.readlines()
# Bic_names = f_Bic.readlines()
# Mot_names = f_Mot.readlines()
#
# sample_bic = []
# Bic_names = sorted(Bic_names)
# for i in range(int(len(Bic_names) / 10)):
#     sample_bic.append(sorted(Bic_names)[i * 10])
#
# sample_mot = []
# Mot_names = sorted(Mot_names)
# for j in range(int(len(Mot_names) / 10)):
#     sample_mot.append(sorted(Mot_names)[j * 10])
#
# final = sorted(list(set(e10_names + sample_bic + sample_mot)))
# print(len(final))
# wr = open('/root/lxq/ECCV2022/Chalearn/together/Train/' + SPLIT + '/train/split/train_v4/all_e10_bic10_mot10_a.txt', 'a')
# for fi in final:
#     wr.write(fi)


def submit2split(pkl_file):
    save_path = pkl_file.replace('.pkl', '')
    create_path(save_path)
    preds = pickle.load(open(pkl_file, 'rb'))
    for k, v in tqdm(preds.items()):
        pickle.dump(v, open(save_path + '/' + k + '.pkl', 'wb'), protocol=4)


submit2split('/root/lxqbigdata/ECCV2022/Chalearn/code/v4/runs/final_test/Month/all_e15/p6_ep100_bs64_img1280_scratch/Month_a_e15_1280_59-99-5_ms_51/submit/0.2/predictions.pkl')