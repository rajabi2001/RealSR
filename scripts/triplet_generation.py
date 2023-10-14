import sys
import os
sys.path.append(".")
from utils import utils_option as option
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor
from deg.two_pipelines_deg import TwoPiplinesDeg
from deg.degradation import PerceptualLoss
import utils.utils_file as utils_file
from torch.nn import functional as F

def correct_size(hr, lr):
    sh = hr.shape[0] / lr.shape[0]
    sw = hr.shape[1] / lr.shape[1]
    sf = max(sh, sw)
    return F.interpolate(lr.unsqueeze(0), scale_factor=sf, mode='bicubic').squeeze()
        
    

if __name__ == '__main__':
    
    config = OmegaConf.load("configs/two_orders_deg.yaml")
    deg = TwoPiplinesDeg(config)
    vgg = PerceptualLoss()

    root_H = config['dataroot_H']
    root_R = config['dataroot_R']
    root_L = root_H.replace('HR','LRS')
    root_D = root_H.replace('HR','DV')

    if not os.path.exists(root_L):
        os.makedirs(root_L)

    if not os.path.exists(root_D):
        os.makedirs(root_D)

    paths_H = utils_file.get_image_paths(root_H)
    assert paths_H, 'Error: H path is empty.'

    
    for p in tqdm(random.sample(paths_H, 10000)):

        img_hr = Image.open(p)
        img_hr =  to_tensor(img_hr).cuda()

        path_R = p.replace("HR","LR")
        img_real = to_tensor(Image.open(path_R)).cuda()
        if img_hr.shape != img_real.shape:
            img_real = correct_size(img_hr, img_real)

        r1_list = []
        for i in range(50):
            img_lr_r1, dv, dd = deg.random_sampling(img_hr)
            fv = vgg(img_real, img_lr_r1).item()
            r1_list.append((img_lr_r1, dv, dd, fv))

        k = 5
        r1_list = sorted(r1_list, key=lambda x: x[-1])
        
        best_lr_r2 = r1_list[0][0]
        best_dv_r2 = r1_list[0][1]
        err_r2 = r1_list[0][-1]

        for i in range(k):
            for j in range(10):
                img_lr_r2, dv = deg.reference_sampling(img_hr, r1_list[i][2], new_deg=True)
                fv = vgg(img_real, img_lr_r2).item()
                if fv < err_r2:
                    err_r2 = fv
                    best_lr_r2 = img_lr_r2
                    best_dv_r2 = dv

        T.ToPILImage()(best_lr_r2).save(p.replace("HR","LRS"))
        torch.save(best_dv_r2, p.replace("HR","DV").replace('png','pth'))
        
        # print('r1:', r1_list[0][-1])
        # print('r2:', err_r2)
        # print()

        # T.ToPILImage()(img_real).show()
        # T.ToPILImage()(best_lr_r2).show()
        

    print("successful")

    