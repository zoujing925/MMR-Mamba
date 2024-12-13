"""
Created: DataConsistency @ Xiyang Cai, 2023/09/09

Data consistency layer for k-space signal.

Ref: DataConsistency in DuDoRNet (https://github.com/bbbbbbzhou/DuDoRNet)

"""

import torch
from torch import nn
from einops import repeat


def data_consistency(k, k0, mask):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    out = (1 - mask) * k + mask * k0
    return out


class DataConsistency(nn.Module):
    """
    Create data consistency operator
    """

    def __init__(self):
        super(DataConsistency, self).__init__()

    def forward(self, k, k0, mask):
        """
        k    - input in frequency domain, of shape (n, nx, ny, 2)
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location  (n, 1, len, 1)
        """

        if k.dim() != 4:  # input is 2D
            raise ValueError("error in data consistency layer!")

        # mask = repeat(mask.squeeze(1, 3), 'b x -> b x y c', y=k.shape[1], c=2)
        mask = torch.tile(mask, (1, mask.shape[2], 1, k.shape[-1]))  ### [n, 320, 320, 2]
        # print("k and k0 shape:", k.shape, k0.shape)
        out = data_consistency(k, k0, mask)

        return out, mask
