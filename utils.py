import numpy as np
import torch


def bright(x, a,b):
    # input datatype np.uint8
    x = np.array(x, dtype='float')
    x = x/(b-a) - 255*a/(b-a)
    x[x>255.0] = 255.0
    x[x<0.0] = 0.0
    x = x.astype(np.uint8)
    return x

def trunc(x):
    # input datatype float
    x[x>255.0] = 255.0
    x[x<0.0] = 0.0
    return x



def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(
        img1 **2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()