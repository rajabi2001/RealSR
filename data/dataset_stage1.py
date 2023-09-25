import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from PIL import Image


class DatasetStage1(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SwinIR-Stage1.
    # If only "paths_H" is provided,R eal-ESRGAN degradation will be done on-the-fly.
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetStage1, self).__init__()
        self.opt = opt

        self.out_size = opt['out_size']
        self.crop_type = opt['crop_type']
        self.use_hflip = opt['use_hflip']
        self.use_rot = opt['use_rot']

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

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
