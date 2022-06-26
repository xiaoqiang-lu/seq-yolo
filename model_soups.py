import os

from models.yolo import Model
import torch
from tqdm import tqdm
from utils.torch_utils import intersect_dicts

def model_soups(weights_path, final_path, cfg):
    models = []
    names = ['Human', 'Bicycle', 'Motorcycle', 'Vehicle']

    for weight in tqdm(weights_path):

        # ckpt = torch.load(weights_path + '/' + weight, map_location='cpu')
        ckpt = torch.load(weight, map_location='cpu')

        model = Model(ckpt['model'].yaml, ch=3, nc=4)
        exclude = []
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)
        models.append(model)

    model_dst = Model(cfg, ch=3, nc=4)
    model_dst.names = names
    model_dst_state_dict = model_dst.state_dict()
    param_name_list = list(model_dst_state_dict.keys())
    print('model soup...')
    for param_name in tqdm(param_name_list):
        param = models[0].state_dict()[param_name]
        for i in range(1, len(models)):
            param += models[i].state_dict()[param_name]
        param = param / len(models)
        model_dst_state_dict[param_name] = param
    model_dst.load_state_dict(model_dst_state_dict, strict=False)
    ckpt = {'model': model_dst}
    torch.save(ckpt, final_path)


PATH = 'runs/train/Day/all_e10_bic10_mot/p6_ep100_bs32_img1280_scratch'
STRUCTURE = 'models/yolov4-p6.yaml'

w_list = [PATH + '/weights/last_059.pt',
          PATH + '/weights/last_069.pt',
          PATH + '/weights/last_079.pt',
          PATH + '/weights/last_089.pt',
          PATH + '/weights/last_099.pt']

model_soups(w_list, PATH + '/59-99-5_ms.pt', STRUCTURE)


