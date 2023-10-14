import sys
sys.path.append(".")
import math
import os
import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util_image
import utils.utils_file as utils_file
import utils.utils_deg as utils_deg
from PIL import Image
import torch
from torch.nn import functional as F
from typing import Sequence
from collections import defaultdict
from torchvision.transforms.functional import to_tensor
from matplotlib import pyplot as plt


class TwoPiplinesDeg():
    '''
    # -----------------------------------------
    # Gets H and then produces degradated H as L.
    # -----------------------------------------
    '''
    def __init__(self, opt):

        self.opt = opt
        self.dataroot_H = opt['dataroot_H']
        self.paths_H = utils_file.get_image_paths(self.dataroot_H )
        assert self.paths_H, 'Error: H path is empty.'

        self.pca_matrices_root = opt['pca_matrices_root']

        self.blur_kernel_size1 = opt['blur_kernel_size1']
        self.kernel_list1 = opt['kernel_list1']
        self.kernel_prob1 = opt['kernel_prob1']
        self.sinc_prob1 = opt['sinc_prob1']
        self.blur_sigma1 = opt['blur_sigma1']
        self.betag_range1 = opt['betag_range1']
        self.betap_range1 = opt['betap_range1']

        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.sinc_prob2 = opt['sinc_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.final_sinc_prob = opt['final_sinc_prob']

        # self.resize_hq = opt['resize_hq']

        # the first degradation
        self.resize_prob1 = opt['resize_prob1']
        self.resize_range1 = opt['resize_range1']
        self.gaussian_noise_prob1 = opt['gaussian_noise_prob1']
        self.noise_range1 = opt['noise_range1']
        self.poisson_scale_range1 = opt['poisson_scale_range1']
        self.gray_noise_prob1 = opt['gray_noise_prob1']
        self.jpeg_range1 = opt['jpeg_range1']

        # the second degradation
        self.resize_prob2 = opt['resize_prob2']
        self.resize_range2 = opt['resize_range2']
        self.gaussian_noise_prob2 = opt['gaussian_noise_prob2']
        self.noise_range2 = opt['noise_range2']
        self.poisson_scale_range2 = opt['poisson_scale_range2']
        self.gray_noise_prob2 = opt['gray_noise_prob2']
        self.jpeg_range2 = opt['jpeg_range2']

        self.scale = opt['scale']
        self.blur_prob = opt['blur_prob']

        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1
        self.jpeger = utils_deg.DiffJPEG(differentiable=False).cuda()

        self.pca_matrices_1 = defaultdict(dict)
        self.pca_matrices_2 = defaultdict(dict)

        for k_size in self.kernel_range:
            # order1
            for k_type in self.kernel_list1:
                pca_matrix_path = self.pca_matrices_root + f"/pipline1/pca_matrix_pipline1_{k_size}_{k_type}.pth"
                self.pca_matrices_1[k_size][k_type] = torch.load(pca_matrix_path, map_location=lambda storage, loc: storage)
            # order2
            for k_type in self.kernel_list2:
                pca_matrix_path = self.pca_matrices_root + f"/pipline2/pca_matrix_pipline2_{k_size}_{k_type}.pth"
                self.pca_matrices_2[k_size][k_type] = torch.load(pca_matrix_path, map_location=lambda storage, loc: storage)

    @torch.no_grad()
    def random_sampling(self, img, return_hr=False):

        deg_vector_1 = []
        deg_vector_2 = []
        deg_dict = {}

        # ------------------------------------
        # Generate kernels (used in the first and second degradation)
        # ------------------------------------

        # First Kernel
        kernel_size_1 = random.choice(self.kernel_range)

        kernel_size_index = (kernel_size_1 - 7) // 2
        kernel_size_onehot = [0, 0, 0, 0, 0, 0, 0, 0]
        kernel_size_onehot[kernel_size_index] = 1
        deg_vector_1 += kernel_size_onehot
        deg_dict['kernel_size_1'] = kernel_size_1
        
        if np.random.uniform() < self.sinc_prob1:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size_1 < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel1 = utils_deg.circular_lowpass_kernel(omega_c, kernel_size_1, pad_to=False)
            
            deg_vector_1 += [omega_c, 0, 0, 0, 0, 0, 0]
            deg_vector_1 += [0] * 15
            deg_dict['kernel_sinc_1'] = True
            deg_dict['kernel_omega_1'] = omega_c
            deg_dict['kernel1'] = kernel1

        else:
            kernel_type_1 = random.choices(self.kernel_list1, self.kernel_prob1)[0]
            kernel1, kernel_type_index_1, sigma_x1, sigma_y1, rotation1, betag1, betap1 = utils_deg.random_mixed_kernels(
                kernel_type_1,
                kernel_size_1,
                self.blur_sigma1,
                self.blur_sigma1, [-math.pi, math.pi],
                self.betag_range1,
                self.betap_range1,
                noise_range=None,
                return_stuff=True
            )

            deg_vector_1 += [0]
            kernel_type_onehot = [0, 0, 0, 0, 0, 0]
            kernel_type_onehot[kernel_type_index_1] = 1
            deg_vector_1 += kernel_type_onehot
            deg_dict['sigma_x1'] = sigma_x1
            deg_dict['sigma_y1'] = sigma_y1
            deg_dict['rotation1'] = rotation1
            deg_dict['betag1'] = betag1
            deg_dict['betap1'] = betap1
            deg_dict['kernel_sinc_1'] = False
            deg_dict['kernel_type_1'] = kernel_type_1
            deg_dict['kernel_type_index_1'] = kernel_type_index_1
            deg_dict['kernel1'] = kernel1

            pca_matrix = self.pca_matrices_1[kernel_size_1][kernel_type_1]
            kernel_code = utils_deg.pca_encode(kernel1, pca_matrix)
            deg_vector_1 += kernel_code.tolist()
        
        # pad kernel
        pad_size = (21 - kernel_size_1) // 2
        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # Second Kernel
        kernel_size_2 = random.choice(self.kernel_range)

        kernel_size_index = (kernel_size_2 - 7) // 2
        kernel_size_onehot = [0, 0, 0, 0, 0, 0, 0, 0]
        kernel_size_onehot[kernel_size_index] = 1
        deg_vector_2 += kernel_size_onehot
        deg_dict['kernel_size_2'] = kernel_size_2

        if np.random.uniform() < self.sinc_prob2:
            if kernel_size_2 < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = utils_deg.circular_lowpass_kernel(omega_c, kernel_size_2, pad_to=False)

            deg_vector_2 += [omega_c, 0, 0, 0, 0, 0, 0]
            deg_vector_2 += [0] * 15
            deg_dict['kernel_sinc_2'] = True
            deg_dict['kernel_omega_2'] = omega_c
            deg_dict['kernel2'] = kernel2

        else:
            kernel_type_2 = random.choices(self.kernel_list2, self.kernel_prob2)[0]
            kernel2, kernel_type_index_2, sigma_x2, sigma_y2, rotation2, betag2, betap2 = utils_deg.random_mixed_kernels(
                kernel_type_2,
                kernel_size_2,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
                return_stuff=True
            )

            deg_vector_2 += [0]
            kernel_type_onehot = [0, 0, 0, 0, 0, 0]
            kernel_type_onehot[kernel_type_index_2] = 1
            deg_vector_2 += kernel_type_onehot
            deg_dict['sigma_x2'] = sigma_x2
            deg_dict['sigma_y2'] = sigma_y2
            deg_dict['rotation2'] = rotation2
            deg_dict['betag2'] = betag2
            deg_dict['betap2'] = betap2
            deg_dict['kernel_sinc_2'] = False
            deg_dict['kernel_type_2'] = kernel_type_2
            deg_dict['kernel_type_index_2'] = kernel_type_index_2
            deg_dict['kernel2'] = kernel2

            pca_matrix = self.pca_matrices_2[kernel_size_2][kernel_type_2]
            kernel_code = utils_deg.pca_encode(kernel2, pca_matrix)
            deg_vector_2 += kernel_code.tolist()

        # pad kernel
        pad_size = (21 - kernel_size_2) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------
        # the final sinc kernel
        # ------------------------------------
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = utils_deg.circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)

            deg_vector_2 += [1]
            kernel_size_index = (kernel_size - 7) // 2
            kernel_size_onehot = [0, 0, 0, 0, 0, 0, 0, 0]
            kernel_size_onehot[kernel_size_index] = 1
            deg_vector_2 += kernel_size_onehot
            deg_vector_2 += [omega_c]
            deg_dict['sinc_pulse'] = False
            deg_dict['sinc_kernel_size'] = kernel_size
            deg_dict['sinc_omega_c'] = omega_c
            deg_dict['kernel_sinc_final'] = sinc_kernel
        else:
            sinc_kernel = self.pulse_tensor
            deg_vector_2 += [0, 0]
            deg_vector_2 += [0, 0, 0, 0, 0, 0, 0, 0]
            deg_dict['sinc_pulse'] = True

        # ------------------------------------
        # [0, 1], BGR to RGB, HWC to CHW
        # ------------------------------------
        if not torch.is_tensor(img):
            img = to_tensor(img)
        img_hq = img.unsqueeze(dim=0).float().cuda()
        kernel1 = torch.FloatTensor(kernel1).cuda()
        kernel2 = torch.FloatTensor(kernel2).cuda()
        sinc_kernel = torch.FloatTensor(sinc_kernel).cuda()
        ori_h, ori_w = img_hq.shape[2:]

        # ------------------------------------
        # The first degradation process
        # ------------------------------------

        # blur
        out = util_image.filter2D(img_hq, kernel1)

        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob1)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range1[1])
            deg_vector_1 += [1, 0, 0, float(scale)]
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range1[0], 1)
            deg_vector_1 += [0, 1, 0, float(scale)]
        else:
            scale = 1
            deg_vector_1 += [0, 0, 1, float(scale)]

        mode = random.choice(["area", "bilinear", "bicubic"])
        if mode == 'area':
            deg_vector_1 += [1, 0, 0]
        elif mode == ' bilinear':
            deg_vector_1 += [0, 1, 0]
        else:
            deg_vector_1 += [0, 0, 1]

        out = F.interpolate(out, scale_factor=scale, mode=mode)
        deg_dict['resize_scale_1'] = scale
        deg_dict['resize_mode_1'] = mode

        # add noise
        if np.random.uniform() < self.gaussian_noise_prob1:
            out, sigma, noise = utils_deg.random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range1, clip=True,
                rounds=False, gray_prob=self.gray_noise_prob1, return_noise=True
            )
            deg_vector_1 += [1, 0, float(sigma/self.noise_range1[1])]
            deg_dict['noise_type_1'] = "gaussian"
            deg_dict['noise_sigma_1'] = sigma
            deg_dict['noise1'] = noise
        else:
            out, noise = utils_deg.random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range1,
                gray_prob=self.gray_noise_prob1,
                clip=True,
                rounds=False,
                return_noise=True
            )
            deg_vector_1 += [0, 1, 1]
            deg_dict['noise_type_1'] = "poisson"
            deg_dict['noise1'] = noise

        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range1)
        deg_dict['jpeg1'] = jpeg_p.clone()
        # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p).cuda()
        deg_vector_1 += [float(jpeg_p)]


        # ------------------------------------
        # The second degradation process
        # ------------------------------------

        # blur
        deg_dict['blur2'] = False
        if np.random.uniform() < self.blur_prob:
            out = util_image.filter2D(out, kernel2)
            deg_dict['blur2'] = True
        
        # select scale of second degradation stage
        if isinstance(self.scale, Sequence):
            min_scale, max_scale = self.scale
            stage2_scale = np.random.uniform(min_scale, max_scale)
        else:
            stage2_scale = self.scale
        stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)
        
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob2)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
            deg_vector_2 += [1, 0, 0, float(scale)]
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
            deg_vector_2 += [0, 1, 0, float(scale)]
        else:
            scale = 1
            deg_vector_2 += [0, 0, 1, scale]

        mode = random.choice(["area", "bilinear", "bicubic"])
        if mode == 'area':
            deg_vector_2 += [1, 0, 0]
        elif mode == ' bilinear':
            deg_vector_2 += [0, 1, 0]
        else:
            deg_vector_2 += [0, 0, 1]

        out = F.interpolate(out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode)
        deg_dict['resize_scale_2'] = scale
        deg_dict['resize_size_2'] = (stage2_h, stage2_w)
        deg_dict['resize_mode_2'] = mode

        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out, sigma, noise = utils_deg.random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True,
                rounds=False, gray_prob=self.gray_noise_prob2, return_noise=True
            )
            deg_vector_2 += [1, 0, float(sigma/self.noise_range2[1])]
            deg_dict['noise_type_2'] = "gaussian"
            deg_dict['noise_sigma_2'] = sigma
            deg_dict['noise2'] = noise
        else:
            out, noise = utils_deg.random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob2,
                clip=True,
                rounds=False,
                return_noise=True
            )
            deg_vector_2 += [0, 1, 1]
            deg_dict['noise_type_2'] = "poisson"
            deg_dict['noise2'] = noise

        # ------------------------------------
        # final degradation
        # ------------------------------------
        if np.random.uniform() < 0.5:
            deg_vector_2 += [0, 1]
            deg_dict['order2'] = 1
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = util_image.filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            deg_dict['jpeg2'] = jpeg_p.clone()
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p).cuda()
        else:
            deg_vector_2 += [1, 0]
            deg_dict['order2'] = 2
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            deg_dict['jpeg2'] = jpeg_p.clone()
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p).cuda()
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = util_image.filter2D(out, sinc_kernel)
        
        deg_vector_2 += [float(jpeg_p)]
        deg_dict['resize_size_final'] = (stage2_h, stage2_w)
        deg_dict['resize_mode_final'] = mode

        # resize back to gt_size since We are doing restoration task
        if stage2_scale != 1:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        
        # [0, 1], float32, rgb, nhwc
        lr = lq.float().contiguous().squeeze()

        dv = torch.cat((torch.tensor(deg_vector_1), torch.tensor(deg_vector_2)), 0)

        if return_hr:
            return img_hq, lr, dv, deg_dict

        return lr, dv, deg_dict

    @torch.no_grad()
    def reference_sampling(self, img, deg_dict, sd=0.05, new_deg=False):

        deg_vector_1 = []
        deg_vector_2 = []

        # ------------------------------------
        # Generate kernels (used in the first and second degradation)
        # ------------------------------------

        # First Kernel
        kernel_size_1 = deg_dict['kernel_size_1']

        kernel_size_index = (kernel_size_1 - 7) // 2
        kernel_size_onehot = [0, 0, 0, 0, 0, 0, 0, 0]
        kernel_size_onehot[kernel_size_index] = 1
        deg_vector_1 += kernel_size_onehot

        
        if deg_dict['kernel_sinc_1']:

            if new_deg:
                omega_c = np.clip(np.random.normal(deg_dict['kernel_omega_1'], sd * ((3*np.pi) / 4)), np.pi/3, np.pi)
                kernel1 = utils_deg.circular_lowpass_kernel(omega_c, kernel_size_1, pad_to=False)
            else:
                kernel1 = deg_dict['kernel1']
            
            deg_vector_1 += [omega_c, 0, 0, 0, 0, 0, 0]
            deg_vector_1 += [0] * 15

        else:
            kernel_type_1 = deg_dict['kernel_type_1']
            kernel_type_index_1 = deg_dict['kernel_type_index_1']

            if new_deg:
                kernel1 = utils_deg.random_mixed_kernels(
                    kernel_type_1,
                    kernel_size_1,
                    (deg_dict['sigma_x1'], sd * (self.blur_sigma1[1] - self.blur_sigma1[0])),
                    (deg_dict['sigma_y1'], sd * (self.blur_sigma1[1] - self.blur_sigma1[0])),
                    (deg_dict['rotation1'], sd * 2 * np.pi),
                    (deg_dict['betag1'], sd * (self.betag_range1[1] - self.betag_range1[0])),
                    (deg_dict['betap1'], sd * (self.betap_range1[1] - self.betap_range1[0])),
                    noise_range=None,
                    ref=True
                )[0]
            else:
                kernel1 = deg_dict['kernel1']

            deg_vector_1 += [0]
            kernel_type_onehot = [0, 0, 0, 0, 0, 0]
            kernel_type_onehot[kernel_type_index_1] = 1
            deg_vector_1 += kernel_type_onehot

            pca_matrix = self.pca_matrices_1[kernel_size_1][kernel_type_1]
            kernel_code = utils_deg.pca_encode(kernel1, pca_matrix)
            deg_vector_1 += kernel_code.tolist()
        
        # pad kernel
        pad_size = (21 - kernel_size_1) // 2
        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # Second Kernel
        kernel_size_2 = deg_dict['kernel_size_2']

        kernel_size_index = (kernel_size_2 - 7) // 2
        kernel_size_onehot = [0, 0, 0, 0, 0, 0, 0, 0]
        kernel_size_onehot[kernel_size_index] = 1
        deg_vector_2 += kernel_size_onehot

        if deg_dict['kernel_sinc_2']:

            if new_deg:
                omega_c = np.clip(np.random.normal(deg_dict['kernel_omega_2'], sd * ((3*np.pi) / 4)), np.pi/3, np.pi)
                kernel2 = utils_deg.circular_lowpass_kernel(omega_c, kernel_size_2, pad_to=False)
            else:
                kernel2 = deg_dict['kernel2']
            
            deg_vector_2 += [deg_dict['kernel_omega_2'], 0, 0, 0, 0, 0, 0]
            deg_vector_2 += [0] * 15

        else:
            kernel_type_2 = deg_dict['kernel_type_2']
            kernel_type_index_2 = deg_dict['kernel_type_index_2']

            if new_deg:
                kernel2 = utils_deg.random_mixed_kernels(
                    kernel_type_2,
                    kernel_size_2,
                    (deg_dict['sigma_x2'], sd * (self.blur_sigma2[1] - self.blur_sigma2[0])),
                    (deg_dict['sigma_y2'], sd * (self.blur_sigma2[1] - self.blur_sigma2[0])),
                    (deg_dict['rotation2'], sd * 2 * np.pi),
                    (deg_dict['betag2'], sd * (self.betag_range2[1] - self.betag_range2[0])),
                    (deg_dict['betap2'], sd * (self.betap_range2[1] - self.betap_range2[0])),
                    noise_range=None,
                    ref=True
                )[0]
            else:
                kernel2 = deg_dict['kernel2']

            deg_vector_2 += [0]
            kernel_type_onehot = [0, 0, 0, 0, 0, 0]
            kernel_type_onehot[kernel_type_index_2] = 1
            deg_vector_2 += kernel_type_onehot

            pca_matrix = self.pca_matrices_2[kernel_size_2][kernel_type_2]
            kernel_code = utils_deg.pca_encode(kernel2, pca_matrix)
            deg_vector_2 += kernel_code.tolist()

        # pad kernel
        pad_size = (21 - kernel_size_2) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------
        # the final sinc kernel
        # ------------------------------------
        if not deg_dict['sinc_pulse']:
            kernel_size = deg_dict['sinc_kernel_size']
            if new_deg:
                omega_c = np.clip(np.random.normal(deg_dict['sinc_omega_c'], sd * ((3*np.pi) / 4)), np.pi/5, np.pi)
                sinc_kernel = utils_deg.circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            else:
                sinc_kernel = deg_dict['kernel_sinc_final']

            deg_vector_2 += [1]
            kernel_size_index = (kernel_size - 7) // 2
            kernel_size_onehot = [0, 0, 0, 0, 0, 0, 0, 0]
            kernel_size_onehot[kernel_size_index] = 1
            deg_vector_2 += kernel_size_onehot
            deg_vector_2 += [deg_dict['sinc_omega_c']]
        else:
            sinc_kernel = self.pulse_tensor
            deg_vector_2 += [0, 0]
            deg_vector_2 += [0, 0, 0, 0, 0, 0, 0, 0]

        # ------------------------------------
        # [0, 1], BGR to RGB, HWC to CHW
        # ------------------------------------
        if not torch.is_tensor(img):
            img = to_tensor(img)
        img_hq = img.unsqueeze(dim=0).float().cuda()
        kernel1 = torch.FloatTensor(kernel1).cuda()
        kernel2 = torch.FloatTensor(kernel2).cuda()
        sinc_kernel = torch.FloatTensor(sinc_kernel).cuda()
        ori_h, ori_w = img_hq.shape[2:]

        # ------------------------------------
        # The first degradation process
        # ------------------------------------

        # blur
        out = util_image.filter2D(img_hq, kernel1)


        # random resize
        scale = deg_dict['resize_scale_1']
        if scale > 1:
            deg_vector_1 += [1, 0, 0, float(scale)]
        elif scale < 1:
            deg_vector_1 += [0, 1, 0, float(scale)]
        else:
            deg_vector_1 += [0, 0, 1, float(scale)]

        mode = deg_dict['resize_mode_1']
        if mode == 'area':
            deg_vector_1 += [1, 0, 0]
        elif mode == ' bilinear':
            deg_vector_1 += [0, 1, 0]
        else:
            deg_vector_1 += [0, 0, 1]

        out = F.interpolate(out, scale_factor=deg_dict['resize_scale_1'], mode=deg_dict['resize_mode_1'])

        # add noise
        if deg_dict['noise_type_1'] == "gaussian":
            if new_deg:
                out, sigma = utils_deg.random_add_gaussian_noise_pt(
                    out, sigma_range=(deg_dict['noise_sigma_1'], sd * (self.noise_range1[1] - self.noise_range1[0])), clip=True,
                    rounds=False, gray_prob=self.gray_noise_prob1, ref=True
                )
            else:
                sigma = deg_dict['noise_sigma_1']
                out = utils_deg.add_noise(out, deg_dict['noise1'].cuda())

            deg_vector_1 += [1, 0, float(sigma/self.noise_range1[1])]

        else:
            out = utils_deg.add_noise(out, deg_dict['noise1'].cuda())
            deg_vector_1 += [0, 1, 1]

        # JPEG compression
        if new_deg:
            jpeg_p = torch.clip(torch.normal(float(deg_dict['jpeg1']), sd * (self.jpeg_range1[1] - self.jpeg_range1[0]), size=(out.size(0),)).to(dtype=out.dtype, device=img.device), 60, 100)
        else:
            jpeg_p = deg_dict['jpeg1']

        # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p).cuda()
        deg_vector_1 += [float(jpeg_p)]


        # ------------------------------------
        # The second degradation process
        # ------------------------------------

        # blur
        if deg_dict['blur2']:
            out = util_image.filter2D(out, kernel2)
        
        # select scale of second degradation stage
        if isinstance(self.scale, Sequence):
            min_scale, max_scale = self.scale
            stage2_scale = np.random.uniform(min_scale, max_scale)
        else:
            stage2_scale = self.scale
        stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)
        
        # random resize
        scale = deg_dict['resize_scale_2']
        if scale > 1:
            deg_vector_2 += [1, 0, 0, float(scale)]
        elif scale < 1:
            deg_vector_2 += [0, 1, 0, float(scale)]
        else:
            scale = 1
            deg_vector_2 += [0, 0, 1, scale]

        mode = deg_dict['resize_mode_2']
        if mode == 'area':
            deg_vector_2 += [1, 0, 0]
        elif mode == ' bilinear':
            deg_vector_2 += [0, 1, 0]
        else:
            deg_vector_2 += [0, 0, 1]
        
        stage2_h, stage2_w = deg_dict['resize_size_2']
        out = F.interpolate(out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode)

        # add noise
        if deg_dict['noise_type_2'] == "gaussian":
            if new_deg:
                out, sigma = utils_deg.random_add_gaussian_noise_pt(
                    out, sigma_range=(deg_dict['noise_sigma_2'], sd * (self.noise_range2[1] - self.noise_range2[0])), clip=True,
                    rounds=False, gray_prob=self.gray_noise_prob2, ref=True
                )
            else:
                sigma = deg_dict['noise_sigma_2']
                out = utils_deg.add_noise(out, deg_dict['noise2'].cuda())

            deg_vector_2 += [1, 0, float(sigma/self.noise_range2[1])]

        else:
            out = utils_deg.add_noise(out, deg_dict['noise2'].cuda())
            deg_vector_2 += [0, 1, 1]


        # ------------------------------------
        # final degradation
        # ------------------------------------
        stage2_h, stage2_w = deg_dict['resize_size_final']
        if deg_dict['order2'] == 1:
            deg_vector_2 += [0, 1]
            # resize back + the final sinc filter
            mode = deg_dict['resize_mode_final']
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = util_image.filter2D(out, sinc_kernel)
            # JPEG compression
            if new_deg:
                jpeg_p = torch.clip(torch.normal(float(deg_dict['jpeg2']), sd * (self.jpeg_range2[1] - self.jpeg_range2[0]), size=(out.size(0),)).to(dtype=out.dtype, device=img.device), 60, 100)
            else:
                jpeg_p = deg_dict['jpeg2']
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p).cuda()
        else:
            deg_vector_2 += [1, 0]
            # JPEG compression
            if new_deg:
                jpeg_p = torch.clip(torch.normal(float(deg_dict['jpeg2']), sd * (self.jpeg_range2[1] - self.jpeg_range2[0]), size=(out.size(0),)).to(dtype=out.dtype, device=img.device), 60, 100)
            else:
                jpeg_p = deg_dict['jpeg2']
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p).cuda()
            # resize back + the final sinc filter
            mode = deg_dict['resize_mode_final']
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = util_image.filter2D(out, sinc_kernel)
        
        deg_vector_2 += [float(jpeg_p)]

        # resize back to gt_size since We are doing restoration task
        if stage2_scale != 1:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        
        # [0, 1], float32, rgb, nhwc
        lr = lq.float().contiguous().squeeze()

        dv = torch.cat((torch.tensor(deg_vector_1), torch.tensor(deg_vector_2)), 0)

        return lr, dv
   
if __name__ == '__main__':
    
    pass
