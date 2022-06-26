import pickle
from tqdm import tqdm
import os
import numpy as np


LENGTH = {'Apr': 71750,
          'Aug': 55281,
          'Jan': 48272,
          'Jul': 19006,
          'Jun': 59618,
          'Mar': 69889,
          'May': 44910,
          'Sep': 184}

def sup_object(mon, preds):
    pred_names, all_names = [], []
    for pred in preds:
        pred_names.append(pred)

    ### change this to your own
    for name in os.listdir('/path/together/Test/split/' + mon + '/images'):
        all_names.append(name.split('.')[0])
    not_names = list(set(all_names) - set(pred_names))
    for not_name in not_names:
        preds[not_name] = {'boxes': [[0, 0, 0, 0]],
                           'labels': [0]}

    return preds


def mon_pkl_hb_seq(path, save_path):
    dict = {}
    for file in tqdm(os.listdir(path)):
       mon = file.split('.')[0]
       m_dict = pickle.load(open(path + '/' + file, 'rb'))
       if len(m_dict) == LENGTH[mon]:
           dict[mon] = m_dict
       else:
           m_dict = sup_object(mon, m_dict)
           dict[mon] = m_dict

    wr = open(save_path, 'wb')
    pickle.dump(dict, wr, protocol=4)


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def score_filter(m_dict, score_threshold):
    new_dict = {}
    for k, v in m_dict.items():

        boxes = v['boxes']
        labels = v['labels']

        scores = np.array(v['scores'])
        max_prop = max(scores)
        if max_prop > score_threshold:
            index = len(scores[scores > score_threshold])
            new_dict[k] = {'boxes': boxes[:index],
                           'labels': labels[:index]}
        else:
            new_dict[k] = {'boxes': [[0, 0, 0, 0]],
                           'labels': [0]}

    return new_dict


def pkl_select_score_hb(path, save_path, score_threshold):
    create_path(save_path)
    dict = {}
    for file in tqdm(os.listdir(path)):
        mon = file.split('_')[0]
        m_dict = pickle.load(open(path + '/' + file, 'rb'))
        m_dict = score_filter(m_dict, score_threshold)

        if len(m_dict) == LENGTH[mon]:
            dict[mon] = m_dict
        else:
            m_dict = sup_object(mon, m_dict)
            dict[mon] = m_dict

    wr = open(save_path + '/predictions.pkl', 'wb')
    pickle.dump(dict, wr, protocol=4)


def check_big_length(path):
    m_dict = pickle.load(open(path, 'rb'))
    for mon in m_dict:
        num = len(m_dict[mon])
        if num == LENGTH[mon]:
            print(mon + ' is ok')
        else:
            print(mon + ' is bad')



path = 'runs/test/Day/a_e5/src_split_seq_0.7_0.3_0.6_0.3_hb'
mon_pkl_hb_seq(path + '/src', path + '/predictions.pkl')
check_big_length(path + '/predictions.pkl')