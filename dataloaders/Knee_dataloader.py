from __future__ import print_function, division
import numpy as np
import pandas as pd
from glob import glob
import random
from skimage import transform

import cv2
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import scipy.io as sio
import torch
import torch.utils.data as data
from .utils import *
from PIL import Image
import matplotlib.pyplot as plt
import pdb
import fastmri
from fastmri.data import transforms as T
from skimage import io
from utils import bright, trunc

import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import pathlib
import h5py
import yaml
import csv
from .transforms import build_transforms

def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path,
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.

    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/home/jc3/Data/",
            brain_path="/home/jc3/Data/",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir


def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class KneeDataset(data.Dataset):
    def __init__(self, kspace_refine, kspace_round, base_dir=None, split='train', MRIDOWN='4X', \
                    SNR=15, transform=None, input_normalize=None, sample_rate=1.0, \
                    MASKTYPE='random', CENTER_FRACTIONS=[0.08], ACCELERATIONS=[4]):
        """
        Args:
            data_dir: data folder for retrieving
                1) Ref: T1 kspace data
                2) Tag: T2 / FLAIR kspace data
        """
        self.kspace_refine = kspace_refine
        self.kspace_round = kspace_round
        self._base_dir = base_dir
        self._MRIDOWN = MRIDOWN
        self.im_ids = []
        self.t2_images = []
        self.t1_undermri_images, self.t2_undermri_images = [], []
        self.mode = split
        
        self.transform = build_transforms(self.mode, MASKTYPE, CENTER_FRACTIONS, ACCELERATIONS)
        self.input_normalize = input_normalize
        
        self.center_fractions = [0.08]          
        self.accelerations = [4]
        
        challenge = 'singlecoil'
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        
        self.cur_path = '/home/sh2/users/zj/code/BRATS_codes/dataloaders/'
        self.csv_file = os.path.join(self.cur_path, "singlecoil_" + self.mode + "_split_less.csv")  # _2volume
        
        self.data_root = '/home/sh2/users/zj/MRI/'
        self.data_path = os.path.join(self.data_root, 'singlecoil_' + self.mode)
        
        self.examples = []

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            for row in reader:
                pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(self.data_path, row[0] + '.h5'))

                pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.data_path, row[1] + '.h5'))

                for slice_id in range(min(pd_num_slices, pdfs_num_slices)):
                    self.examples.append(
                        (os.path.join(self.data_path, row[0] + '.h5'), os.path.join(self.data_path, row[1] + '.h5')
                         , slice_id, pd_metadata, pdfs_metadata, id))
                id += 1

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]
        
        print('Number of images in {}: {:d}'.format(split, len(self.examples)))

    def __getitem__(self, idx):
            # 读取pd
        pd_fname, pdfs_fname, slice, pd_metadata, pdfs_metadata, id = self.examples[idx]

        with h5py.File(pd_fname, "r") as hf:
            pd_kspace = hf["kspace"][slice]

            pd_mask = np.asarray(hf["mask"]) if "mask" in hf else None

            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(pd_metadata)

        if self.transform is None:
            pd_sample = (pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)
        else:
            # print('*****************pd sample*****************')
            pd_sample = self.transform(pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)
        
        with h5py.File(pdfs_fname, "r") as hf:
            pdfs_kspace = hf["kspace"][slice]
            pdfs_mask = np.asarray(hf["mask"]) if "mask" in hf else None

            pdfs_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(pdfs_metadata)

        if self.transform is None:
            pdfs_sample = (pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)
        else:
            # print('*****************pdfs sample*****************')
            pdfs_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)
        
        ############### visualize the data ################
        # fig = plt.figure()
        # plt.axis('off')
        # plt.imshow(pdfs_sample[0], cmap = 'gray')
        # fig.savefig('./test_images/pdfs_sub.png', bbox_inches='tight', pad_inches=0)
        
        # fig = plt.figure()
        # plt.axis('off')
        # plt.imshow(pdfs_sample[1], cmap = 'gray')
        # fig.savefig('./test_images/pdfs_target.png', bbox_inches='tight', pad_inches=0)
        
        
        # sample_stats = {"t1_mean": pd_sample[2], "t1_std": pd_sample[3], "t2_mean": pdfs_sample[2], "t2_std": pdfs_sample[3]}
        # sample = {'image_in': pd_sample[0].unsqueeze(0), 
        #           'image': pd_sample[1].unsqueeze(0), 
        #           'target_in': pdfs_sample[0].unsqueeze(0), 
        #           'target': pdfs_sample[1].unsqueeze(0)
        #           }
        
        # print('pd image min and max:', pd_sample[0].min(), pd_sample[0].max(), 'mean and std:', pd_sample[2], pd_sample[3], 'mean:', pd_sample[0].mean())
        # print('pdfs image min and max:', pdfs_sample[0].min(), pdfs_sample[0].max(), 'mean and std:', pdfs_sample[2], pdfs_sample[3], 'mean:', pdfs_sample[0].mean())
        # print('pd target min and max:', pd_sample[1].min(), pd_sample[1].max())
        # print('pdfs target min and max:', pdfs_sample[1].min(), pdfs_sample[1].max())
        # print('pd target mean and std:', pd_sample[2], pd_sample[3])
        # print('pdfs target mean and std:', pdfs_sample[2], pdfs_sample[3])
        # # print('sample:', sample['image_in'].shape, sample['image'].shape, sample['target_in'].shape, sample['target'].shape)
        
        # return sample, sample_stats
        return (pd_sample, pdfs_sample, id)

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices  

    def __len__(self):
        return len(self.examples)

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
        crop_size = 256
        pad_size = (256-256)//2
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
