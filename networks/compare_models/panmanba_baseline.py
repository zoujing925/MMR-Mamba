import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
import numpy as np
# from mamba_ssm.modules.mamba_simple import Mamba
from .pan_mamba_simple import Mamba
from .pan_refine import Refine
from .ESDR import EDSR, ResBlock
from .fusion_module import Fusion_dynamic


class FreBlock9(nn.Module):
    def __init__(self, channels):
        super(FreBlock9, self).__init__()

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)


    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        amp_fuse = self.amp_fuse(msF_amp)
        amp_fuse = amp_fuse + msF_amp
        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.post(out)
        out = out + x
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return out
    

class FreFusion_RealImaginary(nn.Module):
    def __init__(self, channels):
        super(FreFusion_RealImaginary, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels*2,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels*2,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        # print('msF',msF.shape, msF.dtype, 'panF',panF.shape, panF.dtype)
        # print('msF',msF.real.shape, msF.imag.shape, 'panF',panF.real.shape, panF.imag.shape)
       
        real = self.amp_fuse(torch.cat([msF.real,panF.real],1))
        imag = self.pha_fuse(torch.cat([msF.imag,panF.imag],1))

        out = torch.complex(real, imag)+1e-8
        # print('out',out.shape, out.dtype)
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        # print('out',out.shape, out.dtype)

        return self.post(out)
    
    
class FreFusion(nn.Module):
    def __init__(self, channels):
        super(FreFusion, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))
        
        # amp_fuse = self.amp_fuse(torch.add(msF_amp,panF_amp))
        # pha_fuse = self.pha_fuse(torch.add(msF_pha,panF_pha))

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)
    
  
class FreFusionSum(nn.Module):
    def __init__(self, channels):
        super(FreFusionSum, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)
        

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        # print('msF',msF.shape, msF.dtype, 'panF',panF.shape, panF.dtype)
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        # print('msF_amp',msF_amp.shape, msF_amp.dtype, 'panF_amp',panF_amp.shape, panF_amp.dtype)
        # amp_fuse = self.amp_fuse(torch.add(msF_amp,panF_amp))
        pha_fuse = self.pha_fuse(torch.add(msF_pha,panF_pha))
        amp_fuse = self.amp_fuse(torch.add(msF_amp,panF_amp))
        # amp_fuse = self.amp_fuse(msF_amp)
        # pha_fuse = self.pha_fuse(msF_pha)

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)
    
    
class SE_Block(nn.Module):
    def __init__(self, inchannel, L=8):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, L, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(L, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return y
        

class FreFusionSelectiveSE(nn.Module):
    def __init__(self, channels, M=2, G=32, r=16, stride=1 ,L=8):
        super(FreFusionSelectiveSE, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)
        
        self.se_amp1 = SE_Block(channels, L=8)
        self.se_amp2 = SE_Block(channels, L=8)

    def forward(self, msf, panf):

        batch_size, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        # msF = torch.fft.rfft2(msf+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        # print('msF',msF.shape, msF.dtype, 'panF',panF.shape, panF.dtype)
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        # print('msF_amp',msF_amp.shape, msF_amp.dtype, 'panF_amp',panF_amp.shape, panF_amp.dtype)
        
        ######## select amp ########
        y_msF_amp = self.se_amp1(msF_amp) + 1
        y_panF_amp = self.se_amp2(panF_amp) + 0.5
        msF_amp = msF_amp * y_msF_amp
        panF_amp = panF_amp * y_panF_amp
        
        amp_fuse = self.amp_fuse(torch.add(msF_amp,panF_amp))
        pha_fuse = self.pha_fuse(torch.add(msF_pha,panF_pha))
        
        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)                 
                
class FreFusionSelective(nn.Module):
    def __init__(self, channels, M=2, G=32, r=16, stride=1 ,L=8):
        super(FreFusionSelective, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels*2,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)
        
        self.M = M
        self.features = channels
        d = max(int(channels/r), L)       # 16
        self.d = d
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(channels, d, kernel_size=1, stride=1, bias=False),
                                # nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d,channels*M,1,1,bias=False)
        
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, channels, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, msf, panf):

        batch_size, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        # msF = torch.fft.rfft2(msf+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        # print('msF',msF.shape, msF.dtype, 'panF',panF.shape, panF.dtype)
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        # print('msF_amp',msF_amp.shape, msF_amp.dtype, 'panF_amp',panF_amp.shape, panF_amp.dtype)
        
        ######## select amp ########
        feats = torch.cat([msF_amp, panF_amp], dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        # print('feats',feats.shape)
        feats_U = torch.sum(feats, dim=1)
        # print('feats_U',feats_U.shape)
        feats_S = self.gap(feats_U)
        # print('feats_S',feats_S.shape)    # 4 32 1 1
        feats_Z = self.fc(feats_S)
        # print('feats_Z',feats_Z.shape)
        
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # print('attention_vectors',attention_vectors[0].shape)   # 4 32 1 1
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors) + 0.5
        attention_vectors = attention_vectors[:,1:,:,:,:] + 0.5
        # print('attention_vectors',attention_vectors.shape)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        # print('feats_V',feats_V.shape)
        
        
        # amp_fuse = self.amp_fuse(torch.add(msF_amp,panF_amp))
        amp_fuse = self.amp_fuse(feats_V)
        # pha_fuse = self.pha_fuse(torch.add(msF_pha,panF_pha))
        # amp_fuse = self.amp_fuse(msF_amp)
        pha_fuse = self.pha_fuse(torch.cat(msF_pha,panF_pha))
        
        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)                            
          
          
class FreFusionSelectivePHA(nn.Module):
    def __init__(self, channels, M=2, G=32, r=16, stride=1 ,L=16):
        super(FreFusionSelectivePHA, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)
        
        self.M = M
        self.features = channels
        d = max(int(channels/r), L)       # 16
        self.d = d
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(channels, d, kernel_size=1, stride=1, bias=False),
                                # nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d,channels*M,1,1,bias=False)
        
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, channels, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, msf, panf):

        batch_size, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        # msF = torch.fft.rfft2(msf+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        # print('msF',msF.shape, msF.dtype, 'panF',panF.shape, panF.dtype)
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        # print('msF_amp',msF_amp.shape, msF_amp.dtype, 'panF_amp',panF_amp.shape, panF_amp.dtype)
        
        ######## select pha ########
        feats = torch.cat([msF_pha, panF_pha], dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        # print('feats',feats.shape)
        feats_U = torch.sum(feats, dim=1)
        # print('feats_U',feats_U.shape)
        feats_S = self.gap(feats_U)
        # print('feats_S',feats_S.shape)
        feats_Z = self.fc(feats_S)
        # print('feats_Z',feats_Z.shape)
        
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        # print('attention_vectors',attention_vectors.shape)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        # print('feats_V',feats_V.shape)
        
        
        # amp_fuse = self.amp_fuse(torch.add(msF_amp,panF_amp))
        amp_fuse = self.amp_fuse(msF_amp)
        # pha_fuse = self.pha_fuse(torch.add(msF_pha,panF_pha))
        # amp_fuse = self.amp_fuse(msF_amp)
        pha_fuse = self.pha_fuse(feats_V)
        
        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out) 
    
    
class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, insnorm, threshold):
        insnorm1, insnorm2 = insnorm[0].weight.abs(), insnorm[1].weight.abs()
        # print('insnorm1', insnorm1.max(), insnorm1.min(), insnorm1.mean(), insnorm1.shape)
        # print('insnorm2', insnorm2.max(), insnorm2.min())
        
        insnorm_threshold = insnorm1.min() + 0.05 * (insnorm1.max() - insnorm1.min())
        insnorm_threshold2 = insnorm2.min() + 0.05 * (insnorm2.max() - insnorm2.min())
        # insnorm_threshold = insnorm1.median()
        # print('insnorm_threshold', insnorm_threshold)
        
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        # x1[:, insnorm1 >= insnorm_threshold] = x[0][:, insnorm1 >= insnorm_threshold]
        # x1[:, insnorm1 < insnorm_threshold] = x[1][:, insnorm1 < insnorm_threshold]
        # x2[:, insnorm2 >= insnorm_threshold] = x[1][:, insnorm2 >= insnorm_threshold]
        # x2[:, insnorm2 < insnorm_threshold] = x[0][:, insnorm2 < insnorm_threshold]
               
        # x1[:, insnorm1 >= insnorm_threshold] = x[0][:, insnorm1 >= insnorm_threshold]
        # x1[:, insnorm1 < insnorm_threshold] = x[1][:, insnorm1 < insnorm_threshold] * x[0][:, insnorm1 < insnorm_threshold]
        # x2[:, insnorm2 >= insnorm_threshold] = x[1][:, insnorm2 >= insnorm_threshold]
        # x2[:, insnorm2 < insnorm_threshold] = x[0][:, insnorm2 < insnorm_threshold] * x[1][:, insnorm2 < insnorm_threshold]
               
        x1[:, insnorm1 >= insnorm_threshold] = x[0][:, insnorm1 >= insnorm_threshold]
        x1[:, insnorm1 < insnorm_threshold] = x[1][:, insnorm1 < insnorm_threshold] + x[0][:, insnorm1 < insnorm_threshold]
        x2[:, insnorm2 >= insnorm_threshold2] = x[1][:, insnorm2 >= insnorm_threshold2]
        x2[:, insnorm2 < insnorm_threshold2] = x[0][:, insnorm2 < insnorm_threshold2] + x[1][:, insnorm2 < insnorm_threshold2]
        return [x1, x2]
        # return [x1, x[1]]

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class InstanceNorm2dParallel(nn.Module):
    def __init__(self, num_features):
        super(InstanceNorm2dParallel, self).__init__()
        for i in range(2):
            setattr(self, 'insnorm_' + str(i), nn.InstanceNorm2d(num_features, affine=True, track_running_stats=True))

    def forward(self, x_parallel):
        return [getattr(self, 'insnorm_' + str(i))(x) for i, x in enumerate(x_parallel)]
    
    
class ExchangeBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1,
                 padding=1, activation=False, instance_norm=True):
        super(ExchangeBlock, self).__init__()
        self.conv = ModuleParallel(nn.Conv2d(input_size, output_size, kernel_size, stride, padding))
        self.activation = activation
        self.lrelu = ModuleParallel(nn.LeakyReLU(0.2, True))
        self.instance_norm = instance_norm
        self.insnorm_conv = InstanceNorm2dParallel(output_size)
        self.use_exchange = True
        
        if self.use_exchange:
            self.exchange = Exchange()
            self.insnorm_threshold = 0.02
            self.insnorm_list = []
            for module in self.insnorm_conv.modules():
                if isinstance(module, nn.InstanceNorm2d):
                    self.insnorm_list.append(module)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)
        # print('after conv', out[0].shape)
        if self.instance_norm:
            out = self.insnorm_conv(out)
            if self.use_exchange and len(x) > 1:
                out = self.exchange(out, self.insnorm_list, self.insnorm_threshold)
        return out
    
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms, pan):
        b, c, h, w = ms.shape

        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, ms,pan):
        ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        # print('patch_size',self.patch_size)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        # print('proj',self.proj)
        # print('x befor proj',x.shape)
        x = self.proj(x)
        # print('x after proj',x.shape)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            
        # print('x after flatten',x.shape)
        # x = self.norm(x)
        return x
class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim,bimamba_type=None)
        self.panencoder = Mamba(dim,bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, ms,pan
                ,ms_residual,pan_residual):
        # ms (B,N,C)
        #pan (B,N,C)
        ms_residual = ms+ms_residual
        pan_residual = pan+pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B,N,C = ms.shape
        ms_first_half = ms[:, :, :C//2]
        pan_first_half = pan[:, :, :C//2]
        ms_swap= torch.cat([pan_first_half,ms[:,:,C//2:]],dim=2)
        pan_swap= torch.cat([ms_first_half,pan[:,:,C//2:]],dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap,pan_swap,ms_residual,pan_residual
class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        # print('cross_mamba',self.cross_mamba)
        self.norm1 = LayerNorm(dim,'with_bias')
        # print('norm1',self.norm1)
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        # print('ms_resi',ms_resi.shape)
        ms = self.norm1(ms_resi)
        # print('ms',ms.shape)
        pan = self.norm2(pan)
        # print('pan',pan.shape)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
        # print('global_f',global_f.shape, global_f.transpose(1, 2).shape)
        B,HW,C = global_f.shape
        H = W = np.sqrt(HW).astype(int)
        # ms = global_f.transpose(1, 2).view(B, C, 128*8, 128*8)
        ms = global_f.transpose(1, 2).view(B, C, H, W)
        # print('ms',ms.shape)
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        # print('ms',ms.shape)
        return ms,ms_resi
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi
class Net(nn.Module):
    def __init__(self,num_channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        base_filter=32
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.esdr_encoder = EDSR(n_resblocks=16, n_feats=64, res_scale=1,scale=8, no_upsampling=False, rgb_range=1)
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.embed_dim = base_filter*self.stride*self.patch_size
        
        self.fre1 = FreBlock9(base_filter)
        self.fre2 = FreBlock9(base_filter)
        self.fre3 = FreBlock9(base_filter)
        self.fre4 = FreBlock9(base_filter)
        self.shallow_fusion1 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion3 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion4 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        
        self.ms_to_token = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.ms_to_token2 = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.pan_to_token2 = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.ms_fre_to_token = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.deep_fusion1= CrossMamba(self.embed_dim)
        self.deep_fusion2 = CrossMamba(self.embed_dim)
        self.deep_fusion3 = CrossMamba(self.embed_dim)
        self.deep_fusion4 = CrossMamba(self.embed_dim)
        self.deep_fusion5 = CrossMamba(self.embed_dim)
        self.deep_fusion6 = CrossMamba(self.embed_dim)
        self.deep_fusion7 = CrossMamba(self.embed_dim)
        self.deep_fusion8 = CrossMamba(self.embed_dim)
        self.deep_fusion9 = CrossMamba(self.embed_dim)
        self.deep_fusion10 = CrossMamba(self.embed_dim)
        
        self.cnn_fusion = nn.Sequential(nn.Conv2d(base_filter*2,base_filter*2,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(base_filter,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(base_filter,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),)
        
        self.cnn_fusion0 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter*2,3,1,1),nn.LeakyReLU(0.1,inplace=True))
        self.cnn_fusion1 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter*2,3,1,1),nn.LeakyReLU(0.1,inplace=True))
        self.cnn_fusion2 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(base_filter,base_filter,3,1,1),
                                        nn.Conv2d(base_filter,base_filter,3,1,1))
        self.exchange = ExchangeBlock(base_filter,base_filter)
        self.exchange0 = ExchangeBlock(base_filter,base_filter)
        
        self.cnn1 = nn.Sequential(nn.Conv2d(base_filter,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True))
        self.cnn2 = nn.Sequential(nn.Conv2d(base_filter,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True))
        
        self.fre_cnn_fusion1 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter*2,3,1,1),nn.LeakyReLU(0.1,inplace=True),)
        self.fre_cnn_fusion2 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(base_filter,base_filter,3,1,1),
                                        nn.Conv2d(base_filter,base_filter,3,1,1))
        self.fre_exchange = ExchangeBlock(base_filter,base_filter)
        
        # self.final_fusion = nn.Sequential(nn.Conv2d(base_filter*2,base_filter*2,3,1,1),nn.LeakyReLU(0.1,inplace=True),
        #                     nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
        #                     nn.Conv2d(base_filter,base_filter,3,1,1),
        #                     nn.Conv2d(base_filter,base_filter,3,1,1))
        self.final_fusion0 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True))
        self.final_fusion1 = nn.Sequential(nn.Conv2d(base_filter,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True))
        
        self.fusion_dynamic = Fusion_dynamic(n_feat=32)
        
        self.norm = nn.InstanceNorm2d(base_filter, affine=True)

        self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(6)])
        self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(6)])   
        self.swap_mamba1 = TokenSwapMamba(self.embed_dim)
        self.swap_mamba2 = TokenSwapMamba(self.embed_dim)
        self.patchunembe = PatchUnEmbed(base_filter)
        
        self.fre = FreBlock9(base_filter)
        # self.frefusion = FreFusion(base_filter)
        self.frefusionRI = FreFusion_RealImaginary(base_filter)
        self.frefusionsum = FreFusionSum(base_filter)
        self.frefusionselective = FreFusionSelective(base_filter)
        self.frefusionselectiveSE = FreFusionSelectiveSE(base_filter)
        self.freselectivePHA = FreFusionSelectivePHA(base_filter)
        self.output = Refine(base_filter,1)
        
    def forward(self,ms,pan):
        
        ms_f = self.ms_encoder(ms)
        pan_f = self.pan_encoder(pan)
        # # ms_f = ms_bic
        # # pan_f = pan
        b,c,h,w = ms.shape
        
        # ms_f = self.fre3(ms_f)
        # pan_f = self.fre4(pan_f)

        ms_f = self.ms_to_token(ms_f)
        # print('ms_f', ms_f.shape)
        pan_f = self.pan_to_token(pan_f)
        # print('pan_f', pan_f.shape)
        residual_ms_f = 0
        residual_pan_f = 0
        ms_f,residual_ms_f = self.ms_feature_extraction([ms_f,residual_ms_f])
        pan_f,residual_pan_f = self.pan_feature_extraction([pan_f,residual_pan_f])
        # print('after feature extration', 'ms_f', ms_f.shape, 'residual_ms_f', residual_ms_f.shape, 'pan_f', pan_f.shape, 'residual_pan_f', residual_pan_f.shape)
        # ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba1(ms_f,pan_f,residual_ms_f,residual_pan_f)
        # ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba2(ms_f,pan_f,residual_ms_f,residual_pan_f)
       
        ms_f = self.patchunembe(ms_f,(h,w))
        pan_f = self.patchunembe(pan_f,(h,w))
        # print('after patchunembe', 'ms_f', ms_f.shape, 'pan_f', pan_f.shape)       
        
        # ms_f = self.fre1(ms_f)
        # pan_f = self.fre2(pan_f)
        
        # ms_fre = self.frefusionRI(ms_f,pan_f)
        # ms_fre = self.frefusion(ms_f,pan_f)
        # ms_fre = self.frefusionselective(ms_f,pan_f)
        ms_fre = self.frefusionselectiveSE(ms_f,pan_f)
        # ms_fre = self.freselectivePHA(ms_f,pan_f)
        # ms_fre = self.frefusionsum(ms_f,pan_f)
        # ms_fre = self.cnn_fusion(torch.cat([ms_f,pan_f],dim=1))
        
        # fre_out = self.fre_cnn_fusion1(torch.cat([ms_f,pan_f],dim=1))
        # fre_out1, fre_out2 = torch.chunk(fre_out, 2, dim=1)
        # [fre_out1, fre_out2] = self.fre_exchange([fre_out1, fre_out2])
        # ms_fre = self.fre_cnn_fusion2(torch.cat([fre_out1,fre_out1],dim=1))+ms_f       
        
        # ms_f = self.shallow_fusion1(torch.concat([ms_f,pan_f],dim=1))+ms_f
        # pan_f = self.shallow_fusion2(torch.concat([pan_f,ms_f],dim=1))+pan_f
        # print('after shallow fusion', 'ms_f', ms_f.shape, 'pan_f', pan_f.shape)
        
        # ms_resudiual = ms_f
        
        # ms_f = self.cnn_fusion(torch.cat([ms_f,pan_f],dim=1))+ms_f
        
        ################### half instance norm ###################
        # out = self.cnn_fusion0(torch.cat([ms_f,pan_f],dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        # resi = torch.cat([self.norm(out1), out2], dim=1)
        # out = self.cnn_fusion1(resi)
        # out1, out2 = torch.chunk(out, 2, dim=1)
        # resi = torch.cat([self.norm(out1), out2], dim=1)
        # # ms_f = self.cnn_fusion2(resi)+ms_f
        
        # out1, out2 = torch.chunk(resi, 2, dim=1)
        # ms_f = self.fusion_dynamic(out1, out2)
        # ms_f = self.fusion_dynamic(ms_f,pan_f) + ms_f
        
        ############### 2 channel sum total 7 CNN layers ################
        # ms_f, pan_f = self.exchange0([ms_f, pan_f])
        # out = self.cnn_fusion0(torch.cat([ms_f,pan_f],dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        # [out1, out2] = self.exchange([out1, out2])
        # out = self.cnn_fusion1(torch.cat([out1, out2], dim=1))
        # ms_f = self.cnn_fusion2(out) + ms_f
        
        ############### 2 channel sum total 6 CNN layers ################
        # ms_f, pan_f = self.exchange0([ms_f, pan_f])
        # ms_f, pan_f = self.exchange([ms_f, pan_f])
        # out = self.cnn_fusion1(torch.cat([ms_f, pan_f], dim=1))
        # ms_f = self.cnn_fusion2(out) + ms_f
        
        ############### 2 channel sum total 6 CNN layers ################
        # ms_f, pan_f = self.exchange0([ms_f, pan_f])
        # out = self.cnn_fusion0(torch.cat([ms_f,pan_f],dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        
        # out1 = self.cnn1(ms_f)
        # out2 = self.cnn2(pan_f)
        # [out1, out2] = self.exchange([out1, out2])
        
        # out1 = self.shallow_fusion3(torch.concat([out1, out2],dim=1))+ms_f
        # out2 = self.shallow_fusion4(torch.concat([out1, out2],dim=1))+pan_f
        
        # ms_f = self.cnn_fusion2(torch.cat([out1, out2], dim=1)) + ms_f
        
        ############### No channel total 5 CNN layers ################
        # out = self.cnn_fusion0(torch.cat([ms_f,pan_f],dim=1))
        # out = self.cnn_fusion1(out)
        # ms_f = self.cnn_fusion2(out) + ms_f
        
        # # ######### channel swap half #########
        # out1_half = out1[:, :out1.shape[1]//2, :, :]
        # out2_half = out2[:, :out2.shape[1]//2, :, :]
        # out1 = torch.cat([out1_half, out2[:, out2.shape[1]//2:, :, :]], dim=1)
        # out2 = torch.cat([out2_half, out1[:, out1.shape[1]//2:, :, :]], dim=1)
        
        # ms_f = self.cnn_fusion2(torch.cat([out1,out1],dim=1))+ms_f
        
        
        # ms_f = self.ms_to_token2(ms_f)
        # pan_f = self.pan_to_token2(pan_f)
        # print('after to token', 'ms_f', ms_f.shape, 'pan_f', pan_f.shape)
        residual_ms_f = 0
        residual_fre = 0

        # ms_f,residual_ms_f = self.deep_fusion1(ms_f,residual_ms_f,pan_f)
        # # print('after deep fusion1', 'ms_f', ms_f.shape, 'residual_ms_f', residual_ms_f.shape)
        # ms_f,residual_ms_f = self.deep_fusion2(ms_f,residual_ms_f,pan_f)
        # ms_f,residual_ms_f = self.deep_fusion3(ms_f,residual_ms_f,pan_f)
        # ms_f,residual_ms_f = self.deep_fusion4(ms_f,residual_ms_f,pan_f)      
        
        # f_fre = self.ms_fre_to_token(ms_fre)
        
        # ms_f = self.final_fusion0(torch.cat([ms_f,pan_f],dim=1))
        
        # ms_f,residual_ms_f = self.deep_fusion5(ms_f,residual_ms_f,f_fre)
        # ms_f,residual_ms_f = self.deep_fusion6(ms_f,residual_ms_f,f_fre)
        # ms_f,residual_ms_f = self.deep_fusion7(ms_f,residual_ms_f,f_fre)
        # ms_f,residual_ms_f = self.deep_fusion8(ms_f,residual_ms_f,f_fre)
        # ms_f = self.patchunembe(ms_f,(h,w))
        # print('after patchunembe', 'ms_f', ms_f.shape)
        
        ################ change the order of spatial and frequency ##############
        # f_fre, residual_fre = self.deep_fusion1(f_fre, residual_fre, ms_f)
        # f_fre, residual_fre = self.deep_fusion2(f_fre, residual_fre, ms_f)
        # f_fre, residual_fre = self.deep_fusion3(f_fre, residual_fre, ms_f)
        # f_fre, residual_fre = self.deep_fusion4(f_fre, residual_fre, ms_f)
        # ms_f = self.patchunembe(f_fre,(h,w))
        
        ################ change to 1 spatial 1 frequency, and 2 cross ##############
        # f_fre1, residual_fre = self.deep_fusion1(f_fre, residual_fre, ms_f)
        # f_fre1 = f_fre1 + f_fre
        # ms_f1,residual_ms_f = self.deep_fusion2(ms_f,residual_ms_f,f_fre)
        # ms_f1 = ms_f1 + ms_f
        # ms_f, residual_ms_f = self.deep_fusion3(ms_f1, residual_ms_f, f_fre1)
        # ms_f, residual_ms_f = self.deep_fusion4(ms_f, residual_ms_f, f_fre1)
        # ms_f = self.patchunembe(ms_f,(h,w))
        
        ms_f = self.final_fusion1(ms_fre)
        
        hrms = self.output(ms_f)+ms
        # print('after output', 'hrms', hrms.shape)
        return hrms


def build_model(args):
    return Net(args)