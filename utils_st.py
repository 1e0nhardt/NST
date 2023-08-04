import gc

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from skimage.exposure import match_histograms

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


#! HM
def histogram_matching(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    x_view = content_feat.view(-1, W,H)
    image1_temp = match_histograms(np.array(x_view.detach().clone().cpu().float().transpose(0, 2)),
                                   np.array(style_feat.view(-1, W, H).detach().clone().cpu().float().transpose(0, 2)),
                                   multichannel=True)
    image1_temp = torch.from_numpy(image1_temp).float().to(content_feat.device).transpose(0, 2).view(B, C, W, H)
    return content_feat + (image1_temp - content_feat).detach()


def calc_mean_std(feat, eps=1e-7):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).reshape(N, C, 1, 1)
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


#! wct
def wct(fc_fp32, fs_fp32):
    """
    fc: (1, c, hc, wc)
    fs: (1, c, hs, ws)
    """
    # 转为double计算，提高数值精度
    fc = fc_fp32.to(torch.float64)
    fs = fs_fp32.to(torch.float64)

    b, c, hc, wc = fc.shape

    fc = fc.reshape(b, c, -1)
    fs = fs.reshape(b, c, -1)

    # center
    ms = fs.mean(2, keepdims=True) # (1, c, 1)
    fc = fc - fc.mean(2, keepdims=True)
    fs = fs - ms

    # 先计算协方差阵，协方差阵的奇异值分解就是特征分解。
    # 协方差阵的特征值就是奇异值，是fc的奇异值的平方。特征矩阵就是U=V
    # c_u, c_d, c_v_T = torch.linalg.svd(fc)
    # c_u: (1, c, c)
    # c_d: (1, min(c, hw))
    # c_v_T: (1, hw, hw)
    # 先计算协方差阵的原因是，hw通常远大于c。为了避免出现(hw x hw)的大矩阵，所以先计算协方差。
    # 如果出现数值问题，可以在协方差阵中加上一个单位阵。
    cov_c = torch.bmm(fc, fc.transpose(1,2)).div(fc.shape[1]-1)
    cov_s = torch.bmm(fs, fs.transpose(1,2)).div(fs.shape[1]-1)
    cov_c = torch.nan_to_num(cov_c)
    cov_s = torch.nan_to_num(cov_s)
    Ec, c_e, EcT = torch.linalg.svd(cov_c)
    Es, s_e, EsT = torch.linalg.svd(cov_s)

    k_c = c_e.shape[1]
    for i in range(c_e.shape[1])[::-1]:
        if c_e[0][i] > 1e-5:
            k_c = i + 1
            break
    LOGGER.debug(f'k_c={k_c}')
    # LOGGER.debug(c_e[0][-1])

    # fc_hat = E @ D^-0.5 @ E^T @ fc
    Dc = torch.diagflat(c_e.pow(-0.5)[0][:k_c]).unsqueeze(0) # (1, k_c, k_c)
    step1 = torch.bmm(Ec[:, :, :k_c], Dc) # (1, c, k_c)
    step2 = torch.bmm(step1, EcT[:, :k_c, :]) # (1, c, c)
    fc_hat = torch.bmm(step2, fc) # (1, c, hc x wc)

    k_s = s_e.shape[1]
    for i in range(s_e.shape[1])[::-1]:
        if s_e[0][i] > 1e-5:
            k_s = i + 1
            break
    LOGGER.debug(f'k_s={k_s}')

    Ds = torch.diagflat(s_e.pow(0.5)[0][:k_s]).unsqueeze(0) # (1, k_c, k_c)
    step1 = torch.bmm(Es[:, :, :k_s], Ds) # (1, c, k_c)
    step2 = torch.bmm(step1, EsT[:, :k_s, :]) # (1, c, c)
    fcs_hat = torch.bmm(step2, fc_hat) # (1, c, hc x wc)
    target = (fcs_hat + ms).reshape(b, c, hc, wc)
    return target.to(torch.float32)


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


def calc_remd_loss(A, B, center=True, l2=False):
    # gc.collect()
    # torch.cuda.empty_cache()
    # A,B: [BCHW]
    if not l2:
        C = cosine_dismat(A, B, center)
    else:
        C = l2_dismat(A, B)
    m1, _ = C.min(1) # cosine距离矩阵每列的最小值
    m2, _ = C.min(2) # cosine距离矩阵每行的最小值
    remd = torch.max(m1.mean(), m2.mean())
    # del C
    # gc.collect() # 极其影响速度
    # torch.cuda.empty_cache()
    ##################################测试用###########################################
    # print_gpu_mem()
    # recorder.save_record('dmat_mem_alloc', calc_mem_allocated(C))
    # recorder.save_record('peak_mem_alloc', torch.cuda.memory_allocated() / 1024**2)
    ###################################################################################
    return remd


def l2_dismat(A, B):
    """
    A: (1, 1, h, w)
    B: (1, 1, h, w)
    """
    _,_,h,w=A.shape
    A_mean, A_std = calc_mean_std(A)
    B_mean, B_std = calc_mean_std(B)
    A = (A - A_mean)/A_std
    B = (B - B_mean)/B_std
    A = A.reshape(-1).unsqueeze(1)
    B = B.reshape(-1).unsqueeze(0)
    return (A - B).unsqueeze(0) ** 2


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
    torch.manual_seed(42)
    x = torch.rand(1, 6, 256, dtype=torch.float32)

    # AA^T定实对称矩阵。一定可以被正交对角化。
    co_u, co_d, co_v_T = torch.linalg.svd(torch.bmm(x, x.transpose(1,2)))
    c_u, c_d, c_v_T = torch.linalg.svd(x)
    print(co_d)
    print(co_u.shape)
    print(co_d.shape)
    print(co_v_T.shape)
    print(co_d ** 2)

    print(co_u)
    print(co_v_T)
    # print(torch.allclose(abs(co_u), abs(c_u), atol=1e-7))
    # print(co_u)
    # print(c_u)
    # print(torch.isclose(c_u, co_u))
    # print(torch.bmm(c_u, c_u.transpose(1,2)))
    # c_u[0, :, 4] *=-1
    # print(torch.bmm(c_u, co_v_T))
    exit()

    ret = [0, 0, 0]
    for i in range(1000):
        x = torch.rand(1, 6, 9, dtype=torch.float32)

        # AA^T定实对称矩阵。一定可以被正交对角化。
        c_u, c_d, c_v_T = torch.linalg.svd(torch.bmm(x, x.transpose(1,2)))
        # print(c_u)
        # print(c_v_T)
        if torch.allclose(c_u, c_v_T.transpose(1,2), atol=1e-8): # False
            ret[0] += 1
        if torch.allclose(c_u, c_v_T.transpose(1,2), atol=1e-7): # True
            ret[1] += 1
        if torch.allclose(c_u, c_v_T.transpose(1,2), atol=1e-6): # True
            ret[2] += 1
        # print(torch.bmm(c_u, c_v_T)) # 并不是单位阵，由于数值原因，有约1e-7左右的误差。
    print(ret)
    exit()

    # 默认some=True，返回简化的奇异值分解。 返回U S V
    ret1 = torch.svd(x, some=False) 
    # full_matrices = not some，默认为True。返回U S V^H
    ret2 = torch.linalg.svd(x, full_matrices=True)
    print(ret1[0] == ret2[0])
    print(ret1[1] == ret2[1])
    print(ret1[2] == ret2[2].transpose(1, 2))

    print(torch.eye(4))

