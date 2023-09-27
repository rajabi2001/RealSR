import sys
sys.path.append(".")
import math
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
from deg.degradation import SRMDPreprocessing


class DatasetStage1(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SwinIR-Stage1.
    # If only "paths_H" is provided, Real-ESRGAN degradation will be done on-the-fly.
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetStage1, self).__init__()
        self.opt = opt

        self.dataroot_H = opt['dataroot_H']
        self.out_size = opt['out_size']
        self.crop_type = opt['crop_type']
        self.hflip = opt['hflip']
        self.rotation = opt['rotation']

        self.deg_scale_target = opt['deg_scale_target']
        self.pca_matrix_path = opt['pca_matrix_path']
        self.blur_kernel_size_target = opt['blur_kernel_size_target']
        self.code_length = opt['code_length']
        self.random_kernel_target = opt['random_kernel_target']
        self.noise_target = opt['noise_target']
        self.sig_min_target = opt['sig_min_target']
        self.sig_max_target = opt['sig_max_target']
        self.noise_high_target = opt['noise_high_target']

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

        self.use_sharpener = opt['use_sharpener']
        self.resize_hq = opt['resize_hq']
        self.queue_size = opt['queue_size'] # Queue size of training pool, this should be multiples of batch_size

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
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3

        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1
        self.jpeger = utils_deg.DiffJPEG(differentiable=False)

        # ------------------------------------
        # get paths of H
        # ------------------------------------
        self.paths_H = utils_file.get_image_paths(self.dataroot_H )

        assert self.paths_H, 'Error: H path is empty.'

    @torch.no_grad()
    def __getitem__(self, index):

        # ------------------------------------
        # Generate kernels (used in the first and second degradation)
        # ------------------------------------
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob1:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel1 = utils_deg.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel1 = utils_deg.random_mixed_kernels(
                self.kernel_list1,
                self.kernel_prob1,
                kernel_size,
                self.blur_sigma1,
                self.blur_sigma1, [-math.pi, math.pi],
                self.betag_range1,
                self.betap_range1,
                noise_range=None
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = utils_deg.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = utils_deg.random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------
        # the final sinc kernel
        # ------------------------------------
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = utils_deg.circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # ------------------------------------
        # degradation setup for Target LR
        # ------------------------------------
        pca_matrix_path = self.pca_matrix_path
        pca_matrix = torch.load(pca_matrix_path, map_location=lambda storage, loc: storage)
        deg_process = SRMDPreprocessing(
            scale=self.deg_scale_target, pca_matrix=pca_matrix,
            ksize=self.blur_kernel_size_target, code_length=self.code_length,
            random_kernel=self.random_kernel_target, noise=self.noise_target, cuda=torch.cuda.is_available(), random_disturb=False,
            sig=0, sig_min=self.sig_min_target, sig_max=self.sig_max_target, rate_iso=0.0, rate_cln=0.0, noise_high=self.noise_high_target,
            stored_kernel=False, pre_kernel_path=None
        )

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = Image.open(H_path).convert("RGB")

        # ------------------------------------
        # crop
        # ------------------------------------
        if self.crop_type == "random":
            img_H = util_image.random_crop_arr(img_H, self.out_size)
        elif self.crop_type == "center":
            img_H = util_image.center_crop_arr(img_H, self.out_size)
        # self.crop_type is "none"
        else:
            img_H = np.array(img_H)
            assert img_H.shape[:2] == (self.out_size, self.out_size)

        # ------------------------------------
        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
        # ------------------------------------
        img_hq = (img_H[..., ::-1] / 255.0).astype(np.float32)

        # ------------------------------------
        # flip, rotation
        # ------------------------------------
        img_hq = util_image.augment(img_hq, self.hflip, self.rotation)

        # ------------------------------------
        # [0, 1], BGR to RGB, HWC to CHW
        # ------------------------------------
        img_hq = torch.from_numpy(img_hq[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(dim=0).float()
        kernel1 = torch.FloatTensor(kernel1)
        kernel2 = torch.FloatTensor(kernel2)
        ori_h, ori_w = img_hq.shape[2:]

        # ------------------------------------
        # The target degradation process
        # ------------------------------------
        target_lr, target_deg = deg_process(img_hq, kernel=False) 

        # ------------------------------------
        # The first degradation process
        # ------------------------------------

        # blur
        out = util_image.filter2D(img_hq, kernel1)

        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob1)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range1[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range1[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # add noise
        if np.random.uniform() < self.gaussian_noise_prob1:
            out = utils_deg.random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range1, clip=True,
                rounds=False, gray_prob=self.gray_noise_prob1
            )
        else:
            out = utils_deg.random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range1,
                gray_prob=self.gray_noise_prob1,
                clip=True,
                rounds=False
            )

        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range1)

        # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)


        # ------------------------------------
        # The second degradation process
        # ------------------------------------

        # blur
        if np.random.uniform() < self.blur_prob:
            out = util_image.filter2D(out, kernel2)
        
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
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode
        )
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = utils_deg.random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True,
                rounds=False, gray_prob=self.gray_noise_prob2
            )
        else:
            out = utils_deg.random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob2,
                clip=True,
                rounds=False
            )

        # ------------------------------------
        # two orders
        # ------------------------------------
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = util_image.filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = util_image.filter2D(out, sinc_kernel)

        # resize back to gt_size since We are doing restoration task
        if stage2_scale != 1:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        
        if self.resize_hq and stage2_scale != 1:
            # resize hq
            img_hq = F.interpolate(img_hq, size=(stage2_h, stage2_w), mode="bicubic", antialias=True)
            img_hq = F.interpolate(img_hq, size=(ori_h, ori_w), mode="bicubic", antialias=True)
        
        # [0, 1], float32, rgb, nhwc
        lr = lq.float().contiguous().squeeze()

        # [-1, 1], float32, rgb, nhwc
        # hr = (img_hq * 2 - 1).float().permute(0, 2, 3, 1).contiguous()
        hr = img_hq.float().contiguous().squeeze()

        return {'L': lr, "Target L": target_lr, 'H': hr, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
    
if __name__ == '__main__':
    from omegaconf import OmegaConf
    import torchvision.transforms as T
    config = OmegaConf.load("configs/data_preparation_stage1.yaml")
    mydataset = DatasetStage1(config['dataset'])
    index = 2
    output = mydataset[index]
    hr = output["H"]
    lr = output["L"]
    lr_target = output["Target L"]
    p = output["H_path"]
    T.ToPILImage()(hr).show()
    T.ToPILImage()(lr).show()
    T.ToPILImage()(lr_target).show()
    print("successful")
