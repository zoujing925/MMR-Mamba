# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_sys import VSSM
# from .mamba_sys_vim import VSSM

logger = logging.getLogger(__name__)


class MambaUnet(nn.Module):
    def __init__(self, args, img_size=224, num_classes=1, zero_head=False, vis=False):
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.mamba_unet =  VSSM(
                                patch_size=1,
                                in_chans=1,
                                num_classes=self.num_classes,
                                embed_dim=192,
                                depths=[3, 3, 3, 3],
                                mlp_ratio=4,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                patch_norm=True,
                                use_checkpoint='/home/sh2/users/zj/code/Mamba-UNet/code/pretrained_ckpt/vmamba_tiny_e292.pth')

    def forward(self, x):
        # x = torch.cat([x, x2], dim=1)
        # print("x.size():{}".format(x.size()))
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        # print("x.size() repeat:{}".format(x.size()))
        # print(self.mamba_unet)
        logits = self.mamba_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
            
            
def build_model(args):
    return MambaUnet(args)