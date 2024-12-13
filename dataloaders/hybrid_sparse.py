from __future__ import print_function, division
import numpy as np
from glob import glob
import random
from skimage import transform

import torch
from torch.utils.data import Dataset

class Hybrid(Dataset):

    def __init__(self, base_dir=None, split='train', transform=None):

        super().__init__()
        self._base_dir = base_dir
        self.im_ids = []
        self.images = []
        self.gts = []

        if split=='train':
            self._image_dir = self._base_dir
            imagelist = glob(self._image_dir+"/*_ct.png")
            imagelist=sorted(imagelist)
            for image_path in imagelist:
                gt_path = image_path.replace('ct', 't1')
                self.images.append(image_path)
                self.gts.append(gt_path)


        elif split=='test':
            self._image_dir = self._base_dir
            imagelist = glob(self._image_dir + "/*_ct.png")
            imagelist=sorted(imagelist)
            for image_path in imagelist:
                gt_path = image_path.replace('ct', 't1')
                self.images.append(image_path)
                self.gts.append(gt_path)

        self.transform = transform

        assert (len(self.images) == len(self.gts))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img_in, img, target_in, target= self._make_img_gt_point_pair(index)
        sample = {'image_in': img_in, 'image':img, 'target_in': target_in, 'target': target}
        # print("image in:", img_in.shape)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def _make_img_gt_point_pair(self, index):
        # Read Image and Target

        # the default setting (i.e., rawdata.npz) is 4X64P
        dd = np.load(self.images[index].replace('.png', '_raw_4X64P.npz'))
        # print("images range:", dd['fbp'].max(), dd['ct'].max(), dd['under_t1'].max(), dd['t1'].max())
        _img_in = dd['fbp']
        _img_in[_img_in>0.6]=0.6
        _img_in = _img_in/0.6

        _img = dd['ct']
        _img =(_img/1000*0.192+0.192)
        _img[_img<0.0]=0.0
        _img[_img>0.6]=0.6
        _img = _img/0.6

        _target_in = dd['under_t1']
        _target = dd['t1']

        return _img_in, _img, _target_in, _target

class RandomPadCrop(object):
    def __call__(self, sample):
        new_w, new_h = 400, 400
        crop_size = 384
        pad_size = (400-384)//2
        img_in = sample['image_in']
        img = sample['image']
        target_in = sample['target_in']
        target = sample['target']

        img_in = np.pad(img_in, pad_size, mode='reflect')
        img = np.pad(img, pad_size, mode='reflect')
        target_in = np.pad(target_in, pad_size, mode='reflect')
        target = np.pad(target, pad_size, mode='reflect')

        ww = random.randint(0, np.maximum(0, new_w - crop_size))
        hh = random.randint(0, np.maximum(0, new_h - crop_size))

        img_in = img_in[ww:ww+crop_size, hh:hh+crop_size]
        img = img[ww:ww+crop_size, hh:hh+crop_size]
        target_in = target_in[ww:ww+crop_size, hh:hh+crop_size]
        target = target[ww:ww+crop_size, hh:hh+crop_size]

        sample = {'image_in': img_in, 'image': img, 'target_in': target_in, 'target': target}
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
        img_in = torch.from_numpy(img_in).float()
        img = torch.from_numpy(img).float()
        target_in = torch.from_numpy(target_in).float()
        target = torch.from_numpy(target).float()

        return {'ct_in': img_in,
                'ct': img,
                'mri_in': target_in,
                'mri': target}
