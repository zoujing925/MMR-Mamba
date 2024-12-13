from __future__ import print_function, division
import numpy as np
import pandas as pd
from glob import glob
import random
from skimage import transform
from PIL import Image

import cv2
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from skimage import io
from utils import bright, trunc


def normalize(data, mean, stddev, eps=0.0):
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data (torch.Tensor): Input data to be normalized.
        mean (float): Mean value.
        stddev (float): Standard deviation.
        eps (float, default=0.0): Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.0):
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data (torch.Tensor): Input data to be normalized
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class Hybrid_MYdict(Dataset):

    def __init__(self, kspace_refine, kspace_round, base_dir=None, split='train', MRIDOWN='4X',
                 SNR=15, transform=None, input_normalize=None):

        super().__init__()
        self.kspace_refine = kspace_refine
        self.kspace_round = kspace_round
        self._base_dir = base_dir
        self._MRIDOWN = MRIDOWN
        self.transform = transform
        self.input_normalize = input_normalize
        self.data_dict = {}
        self.data_states_dict = {}

        self.splits_path = "../../MRI/BRATS_100patients/cv_splits_100patients/"

        if split == 'train':
            self.train_file = os.path.join(self.splits_path, 'train_data.csv')
            train_images = pd.read_csv(self.train_file).iloc[:, -1].values.tolist()
            self.t1_images = [image for image in train_images if image.split('_')[-1] == 't1.png']

        elif split == 'test':
            self.test_file = os.path.join(self.splits_path, 'test_data.csv')
            test_images = pd.read_csv(self.test_file).iloc[:, -1].values.tolist()
            self.t1_images = [image for image in test_images if image.split('_')[-1] == 't1.png']

        self.load_data_to_memory()

        # Display stats
        print(f'Number of images in {split}: {len(self.t1_images)}')

    def load_data_to_memory(self):
        for image_path in self.t1_images:
            t2_path = image_path.replace('t1', 't2')

            t1_under_path = image_path
            t2_under_path = image_path.replace('t1', 't2_' + self._MRIDOWN + '_undermri')

            t1_in = np.array(Image.open(os.path.join(self._base_dir, t1_under_path))) / 255.0
            t1 = np.array(Image.open(os.path.join(self._base_dir, image_path))) / 255.0
            t2_in = np.array(Image.open(os.path.join(self._base_dir, t2_under_path))) / 255.0
            t2 = np.array(Image.open(os.path.join(self._base_dir, t2_path))) / 255.0

            if self.input_normalize == "mean_std":
                t1_in, t1_mean, t1_std = normalize_instance(t1_in, eps=1e-11)
                t1 = normalize(t1, t1_mean, t1_std, eps=1e-11)
                t2_in, t2_mean, t2_std = normalize_instance(t2_in, eps=1e-11)
                t2 = normalize(t2, t2_mean, t2_std, eps=1e-11)
                t1_in = np.clip(t1_in, -6, 6)
                t1 = np.clip(t1, -6, 6)
                t2_in = np.clip(t2_in, -6, 6)
                t2 = np.clip(t2, -6, 6)
                sample_stats = {"t1_mean": t1_mean, "t1_std": t1_std, "t2_mean": t2_mean, "t2_std": t2_std}

            elif self.input_normalize == "min_max":
                t1_in = (t1_in - t1_in.min()) / (t1_in.max() - t1_in.min())
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                t2_in = (t2_in - t2_in.min()) / (t2_in.max() - t2_in.min())
                t2 = (t2 - t2.min()) / (t2.max() - t2.min())
                sample_stats = 0

            elif self.input_normalize == "divide":
                sample_stats = 0

            elif self.input_normalize == "None":
                sample_stats = 0

            self.data_dict[image_path] = {
                'image_in': t1_in,
                'image': t1,
                'target_in': t2_in,
                'target': t2
            }
            self.data_states_dict[image_path] = sample_stats

    def __len__(self):
        return len(self.t1_images)

    def __getitem__(self, index):
        image_path = self.t1_images[index]
        sample = self.data_dict[image_path]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.data_states_dict[image_path]
    
        
class Hybrid(Dataset):

    def __init__(self, kspace_refine, kspace_round, base_dir=None, split='train', MRIDOWN='4X', \
                    SNR=15, transform=None, input_normalize=None):

        super().__init__()
        self.kspace_refine = kspace_refine
        self.kspace_round = kspace_round
        self._base_dir = base_dir
        self._MRIDOWN = MRIDOWN
        self.im_ids = []
        self.t2_images = []
        self.t1_undermri_images, self.t2_undermri_images = [], []
        # self.t1_krecon_images, self.t2_krecon_images = [], []
        self.splits_path = "../../MRI/BRATS_100patients/cv_splits_100patients/"

        if split=='train':
            self.train_file = self.splits_path + 'train_data.csv'
            train_images = pd.read_csv(self.train_file).iloc[:, -1].values.tolist()
            self.t1_images = [image for image in train_images if image.split('_')[-1]=='t1.png']
            # print("t1 images:", self.t1_images)

        elif split=='test':
            self.test_file = self.splits_path + 'test_data.csv'
            # self.test_file = self.splits_path + 'train_data.csv'
            test_images = pd.read_csv(self.test_file).iloc[:, -1].values.tolist()
            # test_images = os.listdir(self._base_dir)
            self.t1_images = [image for image in test_images if image.split('_')[-1]=='t1.png']
            # print("t1 images:", self.t1_images)

        
        for image_path in self.t1_images:
            t2_path = image_path.replace('t1', 't2')

            if SNR == 0:
                t1_under_path = image_path

                if self.kspace_refine == "False":
                    t2_under_path = image_path.replace('t1', 't2_' + self._MRIDOWN + '_undermri')

            self.t2_images.append(t2_path)
            self.t1_undermri_images.append(t1_under_path)
            self.t2_undermri_images.append(t2_under_path)

        # print("t1 images:", self.t1_images)
        # print("t2 images:", self.t2_images)
        # print("t1_undermri_images:", self.t1_undermri_images)
        # print("t2_undermri_images:", self.t2_undermri_images)

        self.transform = transform
        self.input_normalize = input_normalize

        assert (len(self.t1_images) == len(self.t2_images))
        assert (len(self.t1_images) == len(self.t1_undermri_images))
        assert (len(self.t1_images) == len(self.t2_undermri_images))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.t1_images)))

    def __len__(self):
        return len(self.t1_images)


    def __getitem__(self, index):

        # t1_in_img = Image.open(self._base_dir + self.t1_undermri_images[index])
        # # print("t1_in_img:", t1_in_img.size)
        # t1_in = np.array(t1_in_img)/255.0
        # print('t1_in:', self._base_dir + self.t1_undermri_images[index])
        t1_in = np.array(Image.open(self._base_dir + self.t1_undermri_images[index]))/255.0
        t1 = np.array(Image.open(self._base_dir + self.t1_images[index]))/255.0
        
        t2_in = np.array(Image.open(self._base_dir + self.t2_undermri_images[index]))/255.0
        t2 = np.array(Image.open(self._base_dir + self.t2_images[index]))/255.0
        
        # print('t1_in:', self._base_dir + self.t1_undermri_images[index])
        # print("images:", t1_in.shape, t1.shape, t2_in.shape, t2.shape)
        # print("t1 before standardization:", t1.max(), t1.min(), t1.mean())
        # save_path = '../test_input/beforeNorm/'
        # t2_img = (t2 * 255).astype(np.uint8)
        # io.imsave(save_path + str(index) + '_t2.png', bright(t2_img,0,0.8))

        if self.input_normalize == "mean_std":
            ### 对input image和target image都做(x-mean)/std的归一化操作
            t1_in, t1_mean, t1_std = normalize_instance(t1_in, eps=1e-11)
            t1 = normalize(t1, t1_mean, t1_std, eps=1e-11)
            t2_in, t2_mean, t2_std = normalize_instance(t2_in, eps=1e-11)
            t2 = normalize(t2, t2_mean, t2_std, eps=1e-11)  

            # t1_krecon = normalize(t1_krecon, t1_mean, t1_std, eps=1e-11)
            # t2_krecon = normalize(t2_krecon, t2_mean, t2_std, eps=1e-11)  

            ### clamp input to ensure training stability.
            t1_in = np.clip(t1_in, -6, 6)
            t1 = np.clip(t1, -6, 6)
            t2_in = np.clip(t2_in, -6, 6) 
            t2 = np.clip(t2, -6, 6)

            # t1_krecon = np.clip(t1_krecon, -6, 6)
            # t2_krecon = np.clip(t2_krecon, -6, 6)
            # print("t1_in after standardization:", t1_in.max(), t1_in.min(), t1_in.mean())
            
            sample_stats = {"t1_mean": t1_mean, "t1_std": t1_std, "t2_mean": t2_mean, "t2_std": t2_std}

        elif self.input_normalize == "min_max":
            t1_in = (t1_in - t1_in.min())/(t1_in.max() - t1_in.min())
            t1 = (t1 - t1.min())/(t1.max() - t1.min())
            t2_in = (t2_in - t2_in.min())/(t2_in.max() - t2_in.min())
            t2 = (t2 - t2.min())/(t2.max() - t2.min())
            sample_stats = 0

        elif self.input_normalize == "divide":
            sample_stats = 0
            
        elif self.input_normalize == "None":
            sample_stats = 0
        
        # save_path = '../test_input/afterNorm/'
        # t2_img = (t2 * 255).astype(np.uint8)
        # io.imsave(save_path + str(index) + '_t2.png', bright(t2_img,0,0.8))


        sample = {'image_in': t1_in, 
                  'image': t1, 
                #   'image_krecon': t1_krecon,
                  'target_in': t2_in, 
                  'target': t2
                #   'target_krecon': t2_krecon
                  }
        
        # print("images shape:", t1_in.shape, t1.shape, t2_in.shape, t2.shape)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, sample_stats



def add_gaussian_noise(img, mean=0, std=1):
    noise = std * torch.randn_like(img) + mean
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0, 1)



class AddNoise(object):
    def __call__(self, sample):
        img_in = sample['image_in']
        img = sample['image']
        target_in = sample['target_in']
        target = sample['target']

        add_gauss_noise = transforms.GaussianBlur(kernel_size=5)
        add_poiss_noise = transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))

        add_noise = transforms.RandomApply([add_gauss_noise, add_poiss_noise], p=0.5)

        img_in = add_noise(img_in)
        target_in = add_noise(target_in)

        sample = {'image_in': img_in, 'image': img, 'target_in': target_in, 'target': target}
        
        return sample




class RandomPadCrop(object):
    def __call__(self, sample):
        new_w, new_h = 256, 256
        crop_size = 240
        pad_size = (256-240)//2
        img_in = sample['image_in']
        img = sample['image']
        target_in = sample['target_in']
        target = sample['target']

        # img_krecon = sample['image_krecon']
        # target_krecon = sample['target_krecon']

        img_in = np.pad(img_in, pad_size, mode='reflect')
        img = np.pad(img, pad_size, mode='reflect')
        target_in = np.pad(target_in, pad_size, mode='reflect')
        target = np.pad(target, pad_size, mode='reflect')

        # img_krecon = np.pad(img_krecon, pad_size, mode='reflect')
        # target_krecon = np.pad(target_krecon, pad_size, mode='reflect')

        ww = random.randint(0, np.maximum(0, new_w - crop_size))
        hh = random.randint(0, np.maximum(0, new_h - crop_size))

        # print("img_in:", img_in.shape)
        img_in = img_in[ww:ww+crop_size, hh:hh+crop_size]
        img = img[ww:ww+crop_size, hh:hh+crop_size]
        target_in = target_in[ww:ww+crop_size, hh:hh+crop_size]
        target = target[ww:ww+crop_size, hh:hh+crop_size]

        # img_krecon = img_krecon[ww:ww+crop_size, hh:hh+crop_size]
        # target_krecon = target_krecon[ww:ww+crop_size, hh:hh+crop_size]

        sample = {'image_in': img_in, 'image': img, \
                  'target_in': target_in, 'target': target}
        return sample


class RandomResizeCrop(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        new_w, new_h = 270, 270
        crop_size = 256
        img_in = sample['image_in']
        img = sample['image']
        target_in = sample['target_in']
        target = sample['target']

        img_in = transform.resize(img_in, (new_h, new_w), order=3)
        img = transform.resize(img, (new_h, new_w), order=3)
        target_in = transform.resize(target_in, (new_h, new_w), order=3)
        target = transform.resize(target, (new_h, new_w), order=3)

        ww = random.randint(0, np.maximum(0, new_w - crop_size))
        hh = random.randint(0, np.maximum(0, new_h - crop_size))

        img_in = img_in[ww:ww+crop_size, hh:hh+crop_size]
        img = img[ww:ww+crop_size, hh:hh+crop_size]
        target_in = target_in[ww:ww+crop_size, hh:hh+crop_size]
        target = target[ww:ww+crop_size, hh:hh+crop_size]

        sample = {'image_in': img_in, 'image': img, 'target_in': target_in, 'target': target}
        return sample



class RandomFlip(object):
    def __call__(self, sample):
        img_in = sample['image_in']
        img = sample['image']
        target_in = sample['target_in']
        target = sample['target']

        # horizontal flip
        if random.random() < 0.5:
            img_in = cv2.flip(img_in, 1)
            img = cv2.flip(img, 1)
            target_in = cv2.flip(target_in, 1)
            target = cv2.flip(target, 1)

        # vertical flip
        if random.random() < 0.5:
            img_in = cv2.flip(img_in, 0)
            img = cv2.flip(img, 0)
            target_in = cv2.flip(target_in, 0)
            target = cv2.flip(target, 0)

        sample = {'image_in': img_in, 'image': img, 'target_in': target_in, 'target': target}
        return sample




class RandomRotate(object):
    def __call__(self, sample, center=None, scale=1.0):
        img_in = sample['image_in']
        img = sample['image']
        target_in = sample['target_in']
        target = sample['target']

        degrees = [0, 90, 180, 270]
        angle = random.choice(degrees)
        
        (h, w) = img.shape[:2]

        if center is None:
            center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)

        img_in = cv2.warpAffine(img_in, matrix, (w, h))
        img = cv2.warpAffine(img, matrix, (w, h))
        target_in = cv2.warpAffine(target_in, matrix, (w, h))
        target = cv2.warpAffine(target, matrix, (w, h))

        sample = {'image_in': img_in, 'image': img, 'target_in': target_in, 'target': target}
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_in = sample['image_in'][:, :, None].transpose((2, 0, 1))
        img = sample['image'][:, :, None].transpose((2, 0, 1))
        target_in = sample['target_in'][:, :, None].transpose((2, 0, 1))
        target = sample['target'][:, :, None].transpose((2, 0, 1))

        # image_krecon = sample['image_krecon'][:, :, None].transpose((2, 0, 1))
        # target_krecon = sample['target_krecon'][:, :, None].transpose((2, 0, 1))

        # print("img_in before_numpy range:", img_in.max(), img_in.min())
        img_in = torch.from_numpy(img_in).float()
        img = torch.from_numpy(img).float()
        target_in = torch.from_numpy(target_in).float()
        target = torch.from_numpy(target).float()

        # image_krecon = torch.from_numpy(image_krecon).float()
        # target_krecon = torch.from_numpy(target_krecon).float()
        # print("img_in range:", img_in.max(), img_in.min())

        return {'image_in': img_in,
                'image': img,
                'target_in': target_in,
                'target': target}
