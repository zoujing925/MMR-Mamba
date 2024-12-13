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

          
class FreFusionSelective(nn.Module):
    def __init__(self, channels, M=2, G=32, r=16, stride=1 ,L=8):
        super(FreFusionSelective, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
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

    def forward(self, t2f, t1f):

        batch_size, _, H, W = t2f.shape
        t2F = torch.fft.rfft2(self.pre1(t2f)+1e-8, norm='backward')
        # t2F = torch.fft.rfft2(t2f+1e-8, norm='backward')
        t1F = torch.fft.rfft2(self.pre2(t1f)+1e-8, norm='backward')
        t2F_amp = torch.abs(t2F)
        t2F_pha = torch.angle(t2F)
        t1F_amp = torch.abs(t1F)
        t1F_pha = torch.angle(t1F)
        
        ######## select amp ########
        feats = torch.cat([t2F_amp, t1F_amp], dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors) + 0.5
        attention_vectors = attention_vectors[:,1:,:,:,:] + 0.5
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        # amp_fuse = self.amp_fuse(torch.add(t2F_amp,t1F_amp))
        amp_fuse = self.amp_fuse(feats_V)
        # pha_fuse = self.pha_fuse(torch.add(t2F_pha,t1F_pha))
        # amp_fuse = self.amp_fuse(t2F_amp)
        pha_fuse = self.pha_fuse(torch.cat([t2F_pha,t1F_pha],1))
        
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

    def forward(self, t2f, t1f):

        batch_size, _, H, W = t2f.shape
        t2F = torch.fft.rfft2(self.pre1(t2f)+1e-8, norm='backward')
        # t2F = torch.fft.rfft2(t2f+1e-8, norm='backward')
        t1F = torch.fft.rfft2(self.pre2(t1f)+1e-8, norm='backward')
        # print('t2F',t2F.shape, t2F.dtype, 't1F',t1F.shape, t1F.dtype)
        t2F_amp = torch.abs(t2F)
        t2F_pha = torch.angle(t2F)
        t1F_amp = torch.abs(t1F)
        t1F_pha = torch.angle(t1F)
        # print('t2F_amp',t2F_amp.shape, t2F_amp.dtype, 't1F_amp',t1F_amp.shape, t1F_amp.dtype)
        
        ######## select pha ########
        feats = torch.cat([t2F_pha, t1F_pha], dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        
        # amp_fuse = self.amp_fuse(torch.add(t2F_amp,t1F_amp))
        amp_fuse = self.amp_fuse(t2F_amp)
        # pha_fuse = self.pha_fuse(torch.add(t2F_pha,t1F_pha))
        # amp_fuse = self.amp_fuse(t2F_amp)
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
        
        insnorm_threshold = insnorm1.min() + 0.05 * (insnorm1.max() - insnorm1.min())
        insnorm_threshold2 = insnorm2.min() + 0.05 * (insnorm2.max() - insnorm2.min())
        # insnorm_threshold = insnorm1.median()
        # print('insnorm_threshold', insnorm_threshold)
        
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
               
        x1[:, insnorm1 >= insnorm_threshold] = x[0][:, insnorm1 >= insnorm_threshold]
        x1[:, insnorm1 < insnorm_threshold] = x[1][:, insnorm1 < insnorm_threshold] * x[0][:, insnorm1 < insnorm_threshold]
        x2[:, insnorm2 >= insnorm_threshold] = x[1][:, insnorm2 >= insnorm_threshold]
        x2[:, insnorm2 < insnorm_threshold] = x[0][:, insnorm2 < insnorm_threshold] * x[1][:, insnorm2 < insnorm_threshold]
               
        # x1[:, insnorm1 >= insnorm_threshold] = x[0][:, insnorm1 >= insnorm_threshold]
        # x1[:, insnorm1 < insnorm_threshold] = x[1][:, insnorm1 < insnorm_threshold] + x[0][:, insnorm1 < insnorm_threshold]
        # x2[:, insnorm2 >= insnorm_threshold2] = x[1][:, insnorm2 >= insnorm_threshold2]
        # x2[:, insnorm2 < insnorm_threshold2] = x[0][:, insnorm2 < insnorm_threshold2] + x[1][:, insnorm2 < insnorm_threshold2]
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
    def __init__(self, dim, ffn_ext1sion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_ext1sion_factor)

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

    def forward(self, t2, t1):
        b, c, h, w = t2.shape

        kv = self.kv_dwconv(self.kv(t1))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(t2))

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
    def __init__(self, dim, num_heads, ffn_ext1sion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_ext1sion_factor, bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, t2,t1):
        t2 = t2+self.cro(self.norm_cro1(t2),self.norm_cro2(t1))
        t2 = t2 + self.ffn(self.norm2(t2))
        return t2


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
        self.t2encoder = Mamba(dim,bimamba_type=None)
        self.t1encoder = Mamba(dim,bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, t2,t1
                ,t2_residual,t1_residual):
        # t2 (B,N,C)
        #t1 (B,N,C)
        t2_residual = t2+t2_residual
        t1_residual = t1+t1_residual
        t2 = self.norm1(t2_residual)
        t1 = self.norm2(t1_residual)
        B,N,C = t2.shape
        t2_first_half = t2[:, :, :C//2]
        t1_first_half = t1[:, :, :C//2]
        t2_swap= torch.cat([t1_first_half,t2[:,:,C//2:]],dim=2)
        t1_swap= torch.cat([t2_first_half,t1[:,:,C//2:]],dim=2)
        t2_swap = self.t2encoder(t2_swap)
        t1_swap = self.t1encoder(t1_swap)
        return t2_swap,t1_swap,t2_residual,t1_residual
class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,t2,t2_resi,t1):
        t2_resi = t2+t2_resi
        t2 = self.norm1(t2_resi)
        t1 = self.norm2(t1)
        global_f = self.cross_mamba(self.norm1(t2),extra_emb=self.norm2(t1))
        # global_f = self.cross_mamba(t2,extra_emb=t1)
        B,HW,C = global_f.shape
        H = W = np.sqrt(HW).astype(int)
        # t2 = global_f.transpose(1, 2).view(B, C, 128*8, 128*8)
        t2 = global_f.transpose(1, 2).view(B, C, H, W)
        t2 =  (self.dwconv(t2)+t2).flatten(2).transpose(1, 2)
        return t2,t2_resi
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
        self.t1_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.t2_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.embed_dim = base_filter*self.stride*self.patch_size
        
        self.shallow_fusion1 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_cat2 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(base_filter,base_filter,3,1,1))
        self.t2_to_token = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.t1_to_token = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.t2_to_token2 = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.t1_to_token2 = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.t2_fre_to_token = PatchEmbed(in_chans=self.base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
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
        self.spa_cat_fusion = nn.Sequential(nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(base_filter,base_filter,3,1,1))
        self.spa_sum_fusion = nn.Sequential(nn.Conv2d(base_filter,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                            nn.Conv2d(base_filter,base_filter,3,1,1))
        
        self.exchange = ExchangeBlock(base_filter,base_filter)
        self.exchange0 = ExchangeBlock(base_filter,base_filter)
        
        self.fre_cnn_fusion1 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter*2,3,1,1),nn.LeakyReLU(0.1,inplace=True),)
        self.fre_cnn_fusion2 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(base_filter,base_filter,3,1,1),
                                        nn.Conv2d(base_filter,base_filter,3,1,1))
        self.fre_exchange = ExchangeBlock(base_filter,base_filter)
        
        self.final_fusion = nn.Sequential(nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                            nn.Conv2d(base_filter,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True),
                            nn.Conv2d(base_filter,base_filter,3,1,1),
                            nn.Conv2d(base_filter,base_filter,3,1,1))
        self.final_fusion0 = nn.Sequential(nn.Conv2d(base_filter*2,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True))
        self.final_fusion1 = nn.Sequential(nn.Conv2d(base_filter,base_filter,3,1,1),nn.LeakyReLU(0.1,inplace=True))
        
        # self.fusion_dynamic = Fusion_dynamic(n_feat=32)
        
        self.norm = nn.InstanceNorm2d(base_filter, affine=True)

        self.t1_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.t2_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])   
        self.patchunembe = PatchUnEmbed(base_filter)
        
        self.frefusionselective = FreFusionSelective(base_filter)
        self.output = Refine(base_filter,1)
        
    def forward(self,t2,t1):
        
        t2_f = self.t2_encoder(t2)
        t1_f = self.t1_encoder(t1)
        # # t2_f = t2_bic
        # # t1_f = t1
        b,c,h,w = t2.shape
        
        t2_f = self.t2_to_token(t2_f)
        t1_f = self.t1_to_token(t1_f)
        residual_t2_f = 0
        residual_t1_f = 0
        t2_f,residual_t2_f = self.t2_feature_extraction([t2_f,residual_t2_f])
        t1_f,residual_t1_f = self.t1_feature_extraction([t1_f,residual_t1_f])
        t2_f = self.patchunembe(t2_f,(h,w))
        t1_f = self.patchunembe(t1_f,(h,w))
        
        
        t2_fre = self.frefusionselective(t2_f,t1_f)    
        
        t2_f = self.shallow_fusion1(torch.concat([t2_f,t1_f],dim=1))+t2_f
        t1_f = self.shallow_fusion2(torch.concat([t1_f,t2_f],dim=1))+t1_f
        # print('after shallow fusion', 't2_f', t2_f.shape, 't1_f', t1_f.shape)
        
        # t2_resudiual = t2_f
        
        ######################## cross mamba in spatial domain ########################
        t2_f = self.t2_to_token2(t2_f)
        t1_f = self.t1_to_token2(t1_f)
        # print('after to token', 't2_f', t2_f.shape, 't1_f', t1_f.shape)
        residual_t2_f = 0
        t2_f,residual_t2_f = self.deep_fusion1(t2_f,residual_t2_f,t1_f)
        t2_f,residual_t2_f = self.deep_fusion2(t2_f,residual_t2_f,t1_f)
        t2_f,residual_t2_f = self.deep_fusion3(t2_f,residual_t2_f,t1_f)
        t2_f,residual_t2_f = self.deep_fusion4(t2_f,residual_t2_f,t1_f)      
        t2_f = self.patchunembe(t2_f,(h,w))
        
        # f_fre = self.t2_fre_to_token(t2_fre)
        
        # t2_f = self.final_fusion0(torch.cat([t2_f,t2_fre],dim=1))
        
        # t2_f,residual_t2_f = self.deep_fusion5(t2_f,residual_t2_f,f_fre)
        # t2_f,residual_t2_f = self.deep_fusion6(t2_f,residual_t2_f,f_fre)
        # t2_f,residual_t2_f = self.deep_fusion7(t2_f,residual_t2_f,f_fre)
        # t2_f,residual_t2_f = self.deep_fusion8(t2_f,residual_t2_f,f_fre)
        # t2_f = self.patchunembe(t2_f,(h,w))
        # print('after patchunembe', 't2_f', t2_f.shape)
        
        ################ change the order of spatial and frequency ##############
        # f_fre, residual_fre = self.deep_fusion1(f_fre, residual_fre, t2_f)
        # f_fre, residual_fre = self.deep_fusion2(f_fre, residual_fre, t2_f)
        # f_fre, residual_fre = self.deep_fusion3(f_fre, residual_fre, t2_f)
        # f_fre, residual_fre = self.deep_fusion4(f_fre, residual_fre, t2_f)
        # t2_f = self.patchunembe(f_fre,(h,w))
        
        ################ change to 1 spatial 1 frequency, and 2 cross ##############
        # residual_fre = 0
        # f_fre1, residual_fre = self.deep_fusion5(f_fre, residual_fre, t2_f)
        # f_fre1 = f_fre1 + f_fre
        # t2_f1,residual_t2_f = self.deep_fusion6(t2_f,residual_t2_f,f_fre)
        # t2_f1 = t2_f1 + t2_f
        # f_fre2, residual_fre = self.deep_fusion7(f_fre1, residual_fre, t2_f1)
        # f_fre2 = f_fre2 + f_fre1
        # t2_f2,residual_t2_f = self.deep_fusion8(t2_f1,residual_t2_f,f_fre1)
        # t2_f2 = t2_f2 + t2_f1
        # # t2_f, residual_t2_f = self.deep_fusion7(t2_f1, residual_t2_f, f_fre1)
        # # t2_f, residual_t2_f = self.deep_fusion8(t2_f, residual_t2_f, f_fre1)
        # t2_f2 = self.patchunembe(t2_f2,(h,w))       
        # f_fre2 = self.patchunembe(f_fre2,(h,w))
        # t2_f = self.final_fusion0(torch.cat([t2_f2,f_fre2],dim=1))
        
        out = self.cnn_fusion0(torch.cat([t2_f,t2_fre],dim=1))
        out = self.cnn_fusion1(out)
        out1, out2 = torch.chunk(out, 2, dim=1)
        [out1, out2] = self.exchange([out1, out2])
        t2_f = self.cnn_fusion2(torch.cat([out1, out2], dim=1)) + t2_f
        
        t2_f = self.final_fusion1(t2_f) + t2_f
        
        hrt2 = self.output(t2_f)+t2
        # print('after output', 'hrt2', hrt2.shape)
        return hrt2


def build_model(args):
    return Net(args)