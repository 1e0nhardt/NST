import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from utils import LOGGER

#! EFDM
def exact_feature_distribution_matching(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    value_content, index_content = torch.sort(content_feat.view(B,C,-1))  # sort conduct a deep copy here.
    value_style, _ = torch.sort(style_feat.view(B,C,-1))  # sort conduct a deep copy here.
    inverse_index = index_content.argsort(-1)
    new_content = content_feat.view(B,C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(B,C,-1).detach())
    return new_content.view(B, C, W, H)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


#! AdaIN
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


#! Increase Entropy
def softmax_smoothing(feats: list, threeD=False):
    func = softmax_smoothing_3d if threeD else softmax_smoothing_2d
    # func = lambda x: x*0.001
    feats[-1] = func(feats[-1])
    feats[-2] = func(feats[-2])
    return feats
    # return list(map(func, feats))


def softmax_smoothing_2d(feats: torch.Tensor, T=1):
    _, _, h, w = feats.shape
    x = rearrange(feats, 'n c h w -> n c (h w)')
    x = x / T
    x = torch.softmax(x, dim=-1)
    return rearrange(x, 'n c (h w) -> n c h w', h=h, w=w)


def softmax_smoothing_3d(feats: torch.Tensor):
    _, c, h, w = feats.shape
    x = rearrange(feats, 'n c h w -> n (c h w)')
    x = torch.softmax(x, dim=-1)
    return rearrange(x, 'n (c h w) -> n c h w', c=c, h=h, w=w)


def get_gaussian_kernel(radius=1, sigma=1):
    k = torch.arange(2*radius +1)
    row = torch.exp( -(((k - radius)/(sigma))**2)/2.)
    out = torch.outer(row, row)
    out = out/torch.sum(out)
    return out


def get_gaussian_conv(chs, radius=1, sigma=1):
    kernel_size = 2*radius + 1
    conv = nn.Conv2d(chs, chs, kernel_size=kernel_size, stride=1, padding=radius, groups=chs, bias=False)
    kernel = get_gaussian_kernel(radius, sigma)
    conv.weight = nn.Parameter(repeat(kernel.clone().detach().reshape(kernel_size, kernel_size), 'h w -> n c h w', n=chs, c=1))
    conv.requires_grad_(False)
    return conv


#! feature replace
def feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, _, _ = b.size()

    assert (n == 1) and (n2 == 1), 'batch_size must be 1'

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []

    # Loop is slow but distance matrix requires a lot of memory
    z_dist = cos_distance(a_flat, b_flat, center=True)

    z_best = torch.argmin(z_dist, 2)
    del z_dist

    z_best = z_best.unsqueeze(1).repeat(1, c, 1)
    feat = torch.gather(b_ref, 2, z_best)

    z_new = feat.view(n, c, h, w)
    return z_new


def cos_distance(a, b, center=True):
    """a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    a_norm = ((a * a).sum(1, keepdims=True) + 1e-8).sqrt()
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()

    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)

    d_mat = 1.0 - torch.matmul(a.transpose(2, 1), b)

    return d_mat


def cosine_dismat(A, B, center=False):
    """
    A: [b, c, h, w],
    B: [b, c, h2, w2]
    -> [b, hw, h2w2]
    """
    A = A.reshape(A.shape[0], A.shape[1], -1)
    B = B.reshape(B.shape[0], B.shape[1], -1)

    if center:
        A = A - A.mean(2, keepdims=True)
        B = B - B.mean(2, keepdims=True)

    # [B,HW] 计算逐像素点的每个通道像素值组成的向量的模。
    A_norm = torch.sqrt((A**2).sum(1)) 
    B_norm = torch.sqrt((B**2).sum(1))

    # 计算cosine-similarity前标准化
    A = (A/A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0,2,1) 
    B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))
    
    # HW,C mul C,HW 逐点计算(每个像素点位置所有通道的像素值构成一个向量)cos距离
    dismat = 1.-torch.bmm(A, B) #! 显存占用大户

    return dismat


def nnst_fs_loss(A, B, center=True):
    """
    A: [b, c, h, w],
    B: [b, c, h2, w2]
    """
    dismat = cosine_dismat(A, B, center)
    d_min, _ = torch.min(dismat, 1)
    return d_min.mean()


def calc_remd_loss(A, B, center=True):
    # A,B: [BCHW]
    C = cosine_dismat(A, B, center)
    m1, _ = C.min(1) # cosine距离矩阵每列的最小值
    m2, _ = C.min(2) # cosine距离矩阵每行的最小值
    remd = torch.max(m1.mean(), m2.mean())
    ##################################测试用###########################################
    # print_gpu_mem()
    # recorder.save_record('dmat_mem_alloc', calc_mem_allocated(C))
    # recorder.save_record('peak_mem_alloc', torch.cuda.memory_allocated() / 1024**2)
    ###################################################################################
    return remd


def split_downsample(x, downsample_factor):
    """
    x: (NCHW) | (CHW)\n
    return [(NChw), ..., (NChw)] 长: downsample_factor**2, 宽高: h = H/downsample_factor
    """
    outputs = []
    for i in range(downsample_factor):
        for j in range(downsample_factor):
            outputs.append(x[..., i::downsample_factor, j::downsample_factor])
    return outputs


if __name__ == '__main__':
    # x = torch.rand(1, 3, 3, 3)
    # print(x)
    # print(x.sum())
    # print(x.shape)

    # print(softmax_smoothing([x])[0].sum())

    print(get_gaussian_kernel(3, 5))
