import os
import pickle
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


pkl_path = 'runs/test/Day/a_e10'
score_threshold = 0.1
for file in os.listdir(pkl_path):

    preds = pickle.load(open(pkl_path + '/' + file, 'rb'))
    m_dict = score_filter(preds, score_threshold)

    wr = open(pkl_path + '/' + file.split('_')[-len(file.split('_'))] + '.pkl', 'wb')
    pickle.dump(m_dict, wr, protocol=4)
