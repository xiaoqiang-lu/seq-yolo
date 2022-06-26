
# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2
import time
import copy
import pickle

CLASSES=('0', '1', '2', '3', '4')
CONF_THRESH = 0.6
NMS_THRESH = 0.3
IOU_THRESH = 0.6
IOU_THRESH_DELETE = 0.3

'''
修改检测结果格式，用作后续处理
第一维：种类
第二维：帧
第三维：bbox
第四维：x1,y1,x2,y2,score
'''


def createLinks(dets_all):
    link_begin=time.time()
    links_all=[]
    #建立每相邻两帧之间的link关系
    frame_num=len(dets_all[0])
    cls_num=len(CLASSES)-1
    #links_all=[] #保存每一类的全部link，第一维为类数，第二维为帧数-1，为该类下的links即每一帧与后一帧之间的link，第三维每帧的box数，为该帧与后一帧之间的link
    for cls_ind in range(cls_num): #第一层循环，类数
        links_cls=[] #保存一类下全部帧的links
        for frame_ind in range(frame_num-1): #第二层循环，帧数-1，不循环最后一帧
            dets1=dets_all[cls_ind][frame_ind]
            dets2=dets_all[cls_ind][frame_ind+1]
            box1_num=len(dets1)
            box2_num=len(dets2)
            #先计算每个box的area
            if frame_ind==0:
                areas1=np.empty(box1_num)
                for box1_ind,box1 in enumerate(dets1):
                    areas1[box1_ind]=(box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
            else: #当前帧的area1就是前一帧的area2，避免重复计算
                areas1=areas2
            areas2=np.empty(box2_num)
            for box2_ind,box2 in enumerate(dets2):
                areas2[box2_ind]=(box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
            #计算相邻两帧同一类的link
            links_frame=[] #保存相邻两帧的links
            for box1_ind,box1 in enumerate(dets1):
                area1=areas1[box1_ind]
                x1=np.maximum(box1[0],dets2[:,0])
                y1=np.maximum(box1[1],dets2[:,1])
                x2=np.minimum(box1[2],dets2[:,2])
                y2=np.minimum(box1[3],dets2[:,3])
                w =np.maximum(0.0, x2 - x1 + 1)
                h =np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                ovrs = inter / (area1 + areas2 - inter)
                links_box=[ovr_ind for ovr_ind,ovr in enumerate(ovrs) if ovr >= IOU_THRESH] #保存第一帧的一个box对第二帧全部box的link
                links_frame.append(links_box)
            links_cls.append(links_frame)
        links_all.append(links_cls)
    link_end=time.time()
    # print('link: {:.4f}s'.format(link_end - link_begin))
    return links_all

def maxPath(dets_all,links_all):
    max_begin=time.time()
    for cls_ind,links_cls in enumerate(links_all):
        dets_cls=dets_all[cls_ind]
        while True:
            rootindex,maxpath,maxsum=findMaxPath(links_cls,dets_cls)
            if len(maxpath) <= 1:
                break
            rescore(dets_cls,rootindex,maxpath,maxsum)
            deleteLink(dets_cls,links_cls,rootindex,maxpath,IOU_THRESH_DELETE)
    max_end=time.time()
    # print('max path: {:.4f}s'.format(max_end - max_begin))

def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def NMS(dets_all):
    for cls_ind,dets_cls in enumerate(dets_all):
        for frame_ind,dets in enumerate(dets_cls):
            keep=nms(dets, NMS_THRESH)
            dets_all[cls_ind][frame_ind]=dets[keep, :]

def findMaxPath(links,dets):
    maxpaths=[] #保存从每个结点到最后的最大路径与分数
    roots=[] #保存所有的可作为独立路径进行最大路径比较的路径
    maxpaths.append([ (box[4],[ind]) for ind,box in enumerate(dets[-1])])
    for link_ind,link in enumerate(links[::-1]): #每一帧与后一帧的link，为一个list
        curmaxpaths=[]
        linkflags=np.zeros(len(maxpaths[0]),int)
        det_ind=len(links)-link_ind-1
        for ind,linkboxes in enumerate(link): #每一帧中每个box的link，为一个list
            if linkboxes == []:
                curmaxpaths.append((dets[det_ind][ind][4],[ind]))
                continue
            linkflags[linkboxes]=1
            prev_ind=np.argmax([maxpaths[0][linkbox][0] for linkbox in linkboxes])
            prev_score=maxpaths[0][linkboxes[prev_ind]][0]
            prev_path=copy.copy(maxpaths[0][linkboxes[prev_ind]][1])
            prev_path.insert(0,ind)
            curmaxpaths.append((dets[det_ind][ind][4]+prev_score,prev_path))
        root=[maxpaths[0][ind] for ind,flag in enumerate(linkflags) if flag == 0]
        roots.insert(0,root)
        maxpaths.insert(0,curmaxpaths)
    roots.insert(0,maxpaths[0])
    maxscore=0
    maxpath=[]
    for index,paths in enumerate(roots):
        if paths==[]:
            continue
        maxindex=np.argmax([path[0] for path in paths])
        if paths[maxindex][0]>maxscore:
            maxscore=paths[maxindex][0]
            maxpath=paths[maxindex][1]
            rootindex=index
    return rootindex,maxpath,maxscore

def rescore(dets, rootindex, maxpath, maxsum):
    newscore=maxsum/len(maxpath)
    for i,box_ind in enumerate(maxpath):
        dets[rootindex+i][box_ind][4]=newscore

def deleteLink(dets,links, rootindex, maxpath,thesh):
    for i,box_ind in enumerate(maxpath):
        areas=[(box[2]-box[0]+1)*(box[3]-box[1]+1) for box in dets[rootindex+i]]
        area1=areas[box_ind]
        box1=dets[rootindex+i][box_ind]
        x1=np.maximum(box1[0],dets[rootindex+i][:,0])
        y1=np.maximum(box1[1],dets[rootindex+i][:,1])
        x2=np.minimum(box1[2],dets[rootindex+i][:,2])
        y2=np.minimum(box1[3],dets[rootindex+i][:,3])
        w =np.maximum(0.0, x2 - x1 + 1)
        h =np.maximum(0.0, y2 - y1 + 1)
        inter = w * h
        ovrs = inter / (area1 + areas - inter)
        deletes=[ovr_ind for ovr_ind,ovr in enumerate(ovrs) if ovr >= thesh] #保存待删除的box的index
        for delete_ind in deletes:
            if delete_ind!=box_ind:
                dets[rootindex+i][delete_ind, 4] = 0
        if rootindex+i<len(links): #除了最后一帧，置box_ind的box的link为空
            for delete_ind in deletes:
                links[rootindex+i][delete_ind]=[]
        if i > 0 or rootindex>0:
            for priorbox in links[rootindex+i-1]: #将前一帧指向box_ind的link删除
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)


def dinms(dets, do_snms=True):

    if do_snms:
        links=createLinks(dets)
        maxPath(dets,links)
    NMS(dets)
    boxes=[[] for i in dets[0]]
    classes=[[] for i in dets[0]]
    scores=[[] for i in dets[0]]

    for cls_id, det_cls in enumerate(dets):
        for frame_id, frame in enumerate(det_cls):
            for box_id, box in enumerate(frame):
                if box[4] >= CONF_THRESH:
                    ymin = box[1]
                    xmin = box[0]
                    ymax = box[3]
                    xmax = box[2]
                    boxes[frame_id].append(np.array([ymin, xmin, ymax, xmax]))
                    classes[frame_id].append(cls_id+1)
                    scores[frame_id].append(box[4])
    return boxes, classes, scores




# dets = [[] for i in CLASSES[1:]]
#
# preds = pickle.load(open('./video_pred.pkl', 'rb'))
#
# for cls_ind, cls in enumerate(CLASSES[1:]):
#     for k, v in preds.items():
#         single_img_num_boxes = len(v['boxes'])
#         cls_boxes = np.zeros((single_img_num_boxes, 4), dtype=np.float64)
#         cls_scores = np.zeros((single_img_num_boxes, 1), dtype=np.float64)
#         for box_ind, box in enumerate(v['boxes']):
#             cls_boxes[box_ind][0] = box[1]
#             cls_boxes[box_ind][1] = box[0]
#             cls_boxes[box_ind][2] = box[3]
#             cls_boxes[box_ind][3] = box[2]
#             # cls_boxes[box_ind][:] = box[:]
#
#             if str(v['labels'][box_ind] + 1) == cls:
#                 cls_scores[box_ind][0] = v['scores'][box_ind]
#             else:
#                 cls_scores[box_ind][0] = 0.00001
#         cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float64)
#         dets[cls_ind].append(cls_dets)
#
#
# boxes, classes, scores = dinms(dets, True)
#
# new_dict = {}
# for ind, (k, v) in enumerate(preds.items()):
#     b = [list(j) for j in boxes[ind]]
#     c = [i - 1 for i in classes[ind]]
#     new_dict[k] = {'boxes': b,
#                    'labels': c}
#
# pickle.dump(new_dict, open('./video_pred_seqnms.pkl', 'wb'), protocol=4)



from tqdm import tqdm

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def VN(mon):
    path = '/root/lxq/ECCV2022/Chalearn/Test/' + mon + '/frames'
    video_names = []
    for b in os.listdir(path):
        for s in os.listdir(path + '/' + b):
            video_name = b + '_' + s
            video_names.append(video_name)

    return video_names

def seqnms(path, mon, save_path):

    create_path(save_path)
    final_dict = {}
    preds = pickle.load(open(path, 'rb'))
    v_names = VN(mon)
    for video_name in tqdm(v_names):
        one_video = {}
        for k, v in preds.items():
            if k.startswith(video_name):
                one_video[k] = v
        if one_video:
            preds = dict(sorted(one_video.items(), key=lambda x:x[0]))

            dets = [[] for i in CLASSES[1:]]
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

            for m, n in new_dict.items():
                if not n['boxes']:
                    n['boxes'] = [[0, 0, 0, 0]]
                    n['labels'] = [0]
            final_dict.update(new_dict)

    pickle.dump(final_dict, open(save_path + '/' + mon + '_seq_nms.pkl', 'wb'), protocol=4)


# [Apr, Aug, Jan, Jul, Jun, Mar, May, Sep]
# MONTH = 'Aug'
# seqnms('runs/final_test/Day/all_every5frames/p7_ep100_bs16_img1536_scratch/Day_a_e5_v4p7_1536_29-99-8_ms_0.5_0.1/src/' + MONTH + '_1536_29_39_49_59_69_79_89_99_ms_0.5_0.1.pkl',
#        MONTH,
#        'runs/final_test/Day/all_every5frames/p7_ep100_bs16_img1536_scratch/Day_a_e5_v4p7_1536_29-99-8_ms_0.5_0.1/src_seq')


# MONTH = 'May'    # [Apr, Aug, Jan, Jul, Jun, Mar, May, Sep]
# path = 'runs/final_test/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch/Day_a_e10_bic10_mot_1280_59-99-5_ms_51/src_split/' + MONTH
# save_path = path.replace('src_split', 'src_split_seq')
# create_path(save_path)
# for file in tqdm(os.listdir(path)):
#     dets = [[] for i in CLASSES[1:]]
#     preds = pickle.load(open(path + '/' + file, 'rb'))
#     for cls_ind, cls in enumerate(CLASSES[1:]):
#         for k, v in preds.items():
#             single_img_num_boxes = len(v['boxes'])
#             cls_boxes = np.zeros((single_img_num_boxes, 4), dtype=np.float64)
#             cls_scores = np.zeros((single_img_num_boxes, 1), dtype=np.float64)
#             for box_ind, box in enumerate(v['boxes']):
#                 cls_boxes[box_ind][0] = box[1]
#                 cls_boxes[box_ind][1] = box[0]
#                 cls_boxes[box_ind][2] = box[3]
#                 cls_boxes[box_ind][3] = box[2]
#                 # cls_boxes[box_ind][:] = box[:]
#
#                 if str(v['labels'][box_ind] + 1) == cls:
#                     cls_scores[box_ind][0] = v['scores'][box_ind]
#                 else:
#                     cls_scores[box_ind][0] = 0.00001
#             cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float64)
#             dets[cls_ind].append(cls_dets)
#
#
#     boxes, classes, scores = dinms(dets, True)
#
#     new_dict = {}
#     for ind, (k, v) in enumerate(preds.items()):
#         b = [list(j) for j in boxes[ind]]
#         c = [i - 1 for i in classes[ind]]
#         new_dict[k] = {'boxes': b,
#                        'labels': c}
#
#     pickle.dump(new_dict, open(save_path + '/' + file, 'wb'), protocol=4)