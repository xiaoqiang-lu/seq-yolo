import shutil
from tqdm import tqdm
import os

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


### change the path to your own
PATH = 'your own path'
create_testset(PATH +'/Test',
               PATH +'/together/Test/split')