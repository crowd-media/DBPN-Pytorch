


from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn.dbpn import Net as DBPN
from dbpn.dbpn_v1 import Net as DBPNLL
from dbpn.dbpn_iterative import Net as DBPNITER
from data import get_eval_set
from functools import reduce

# from scipy.misc import imsave
import scipy.io as sio
import time
import cv2




def print_version():
    print('DBPN VERSION 0.1.0')



def _load(checkpoint_path, device = "cuda"):
    if device == "cuda":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint



def load_model(path, device = "cuda", model_type='DBPNLL', upscale_factor=8):

    print('===> Building model')
    # if model_type == 'DBPNLL':
    #     model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=upscale_factor) ###D-DBPN
    # elif model_type == 'DBPN-RES-MR64-3':
    #     model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=upscale_factor) ###D-DBPN
    # else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=upscale_factor) ###D-DBPN
        
    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids=gpus_list)

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, device)

    # print(checkpoint.items())
    s = checkpoint
    # s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        # print(k)
        new_s[k.replace("module.", "")] = v
    
    model.load_state_dict(new_s)

    # print(model)



    # model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    # print('Pre-trained SR model is loaded.')

    if device == "cuda":
        model = model.cuda(gpus_list[0])
    else:
        model.to(device)
    
    return model.eval()




def create_data_loader(dataset_path, upscale_factor, test_batch_size, threads = 1):
    test_set = get_eval_set(dataset_path, upscale_factor)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=test_batch_size, shuffle=False)
    return testing_data_loader



def save_img(img, img_name, save_dir):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    # save img
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])