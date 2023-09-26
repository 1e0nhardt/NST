import gc
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from skimage.exposure import match_histograms
from torchvision import transforms

from utils import LOGGER

###############################################################################
######################### Segment Mask Tools ##################################
###############################################################################
def get_seg_dicts(cont_feats, styl_feats, cmask, smask):
    """
    B=1
    :param cont_feat: List([N, cH1, cW1], ...)
    :param styl_feat: List([N, sH1, sW1], ...)
    :param cmask: [h, w]
    :param smask: [h, w]
    :return content_mask_dict, style_feat_dict
    """
    content_mask_dict = {}
    style_feat_dict = {}

    for content_feat, style_feat in zip(cont_feats, styl_feats):
        B, N, cH, cW = content_feat.shape
        _, _, sH, sW = style_feat.shape

        offset_a = 0
        offset_b = 0

        # 如果特征图宽高太大，则进行稀疏采样
        if max(cH, cW) > 128:
            stride = max(cH, cW) // 128
            content_feat = content_feat[:, :, offset_a::stride, offset_b::stride]
            _, _, cH, cW = content_feat.shape
        
        if max(sH, sW) > 128:
            stride = max(sH, sW) // 128
            style_feat = style_feat[:, :, offset_a::stride, offset_b::stride]
            #! center style features
            style_feat = style_feat - style_feat.mean(2, keepdims=True)
            _, _, sH, sW = style_feat.shape

        # content_feat = content_feat.reshape(B, N, -1)[0]
        style_feat = style_feat.reshape(B, N, -1)[0]

        # 找有效的label set。
        label_set, label_indicator = compute_label_info(cmask, smask)

        # 将mask resize到当前特征图大小。 interpolate要求输入为四维或更高 (N,C,H,W)
        resized_content_segment = F.interpolate(cmask[None, None, :, :], (cH, cW), mode='nearest').squeeze()
        resized_style_segment = F.interpolate(smask[None, None, :, :], (sH, sW), mode='nearest').squeeze()

        for label in label_set: # 逐标签处理
            if not label_indicator[label]: # 判断标签是否可用
                continue
            
            # 找到mask对应的索引
            # 这种用法返回满足条件的元素的坐标位置。注意返回的是这样的元组:(indices,)
            content_mask_index = torch.where(resized_content_segment.reshape(-1) == label)[0].to('cuda')
            style_mask_index = torch.where(resized_style_segment.reshape(-1) == label)[0].to('cuda')
            if content_mask_index is None or style_mask_index is None:
                continue
            
            if content_mask_dict.get(label) is None:
                content_mask_dict[label] = [content_mask_index]
            else:
                content_mask_dict[label].append(content_mask_index)
            
            # 根据索引找到mask对应的特征
            masked_style_feat = torch.index_select(style_feat, 1, style_mask_index)
            if style_feat_dict.get(label) is None:
                style_feat_dict[label] = [masked_style_feat]
            else:
                style_feat_dict[label].append(masked_style_feat)
            
    return content_mask_dict, style_feat_dict

def compute_label_info(cont_seg, styl_seg):
    """
    统计有多少有效label
    :param cont_seg (h, w) 
    :param styl_seg (h, w)
    :return label_set, label_valid_indicator
    """
    if cont_seg.size is False or styl_seg.size is False:
        return
    max_label = torch.max(cont_seg) + 1
    label_set = torch.unique(cont_seg)
    label_indicator = torch.zeros(max_label)
    for l in label_set:
        # if l==0:
        #   continue
        is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100 # 两个mask不能太小，两者之间像素数量差距也不要差100倍以上
        o_cont_mask = torch.where(cont_seg.reshape(-1) == l)[0]
        o_styl_mask = torch.where(styl_seg.reshape(-1) == l)[0]
        label_indicator[l] = is_valid(o_cont_mask.numel(), o_styl_mask.numel())
    # 索引类型为uint8会出现奇怪的行为 ==> 对一维数组索引，得到一个二维的数组。 
    # tensor作索引会出现奇怪的行为
    return [l.item() for l in label_set.long()], [l.item() for l in label_indicator.long()]

def load_segment(image_path, size=None):
    """
    Transfer color labels to number labels
    :param image_path
    :param size (w, h)
    :return tensor (h, w)
    """
    def change_seg(seg):
        color_dict = {
            (153, 45, 165): 3,  # blue
            (45, 98, 166): 2,  # blue
            (0, 0, 0): 0,  # black
            (255, 255, 255): 1,  # white
            (165, 45, 46): 4,  # red
            (165, 139, 44): 5,  # yellow
            (128, 128, 128): 6,  # grey
            (0, 255, 255): 7,  # lightblue
            (255, 0, 255): 8  # purple
        }
        # color_dict = {
        #     (0, 0, 255): 3,  # blue
        #     (0, 255, 0): 2,  # green
        #     (0, 0, 0): 0,  # black
        #     (255, 255, 255): 1,  # white
        #     (255, 0, 0): 4,  # red
        #     (255, 255, 0): 5,  # yellow
        #     (128, 128, 128): 6,  # grey
        #     (0, 255, 255): 7,  # lightblue
        #     (255, 0, 255): 8  # purple
        # }
        arr_seg = np.array(seg) # (h, w, 3)
        new_seg = np.zeros(arr_seg.shape[:-1]) # (h, w)
        for x in range(arr_seg.shape[0]):
            for y in range(arr_seg.shape[1]):
                if tuple(arr_seg[x, y, :]) in color_dict:
                    new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
                else:
                    min_dist_index = 0
                    min_dist = 99999
                    for key in color_dict:
                        dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_index = color_dict[key]
                        elif dist == min_dist:
                            try:
                                min_dist_index = new_seg[x, y - 1, :]
                            except Exception:
                                pass
                    new_seg[x, y] = min_dist_index
        return new_seg.astype(np.uint8)

    if not os.path.exists(image_path):
        print("Can not find segment image path: %s " % image_path)
        return None

    image = Image.open(image_path).convert("RGB")

    if size is not None:
        image.resize(size, Image.NEAREST)

    image = np.array(image)
    # print(image.shape)
    # vr, cr = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    # for v, c in zip(vr, cr):
    #     if c > 500:
    #         print(f'{v}: {c}')
    image = change_seg(image)

    # image *= 48
    # Image.fromarray(image, 'L').save('loaded_seg.png')
    # print(image.shape)
    # print(torch.from_numpy(image).shape)
    return torch.from_numpy(image)
###############################################################################
######################### Segment Mask Tools end ##############################
###############################################################################

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
                                   channel_axis=0)
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

###############################################################################
###################################cWCT########################################
###############################################################################
def cholesky_dec(conv, invert=False):
        cholesky = torch.linalg.cholesky if torch.__version__ >= '1.8.0' else torch.cholesky
        try:
            L = cholesky(conv)
        except RuntimeError:
            # print("Warning: Cholesky Decomposition fails")
            iden = torch.eye(conv.shape[-1]).to(conv.device)
            eps = 1e-8
            while True:
                try:
                    conv = conv + iden * eps
                    L = cholesky(conv)
                    break
                except RuntimeError:
                    eps = eps+1e-8

        if invert:
            L = torch.inverse(L)

        return L.to(conv.dtype)

def whitening(x):
    mean = torch.mean(x, -1)
    mean = mean.unsqueeze(-1).expand_as(x)
    x = x - mean

    conv = (x @ x.transpose(-1, -2)).div(x.shape[-1] - 1)
    inv_L = cholesky_dec(conv, invert=True)

    whiten_x = inv_L @ x

    return whiten_x

def coloring(whiten_xc, xs):
    xs_mean = torch.mean(xs, -1)
    xs = xs - xs_mean.unsqueeze(-1).expand_as(xs)

    conv = (xs @ xs.transpose(-1, -2)).div(xs.shape[-1] - 1)
    Ls = cholesky_dec(conv, invert=False)

    coloring_cs = Ls @ whiten_xc
    coloring_cs = coloring_cs + xs_mean.unsqueeze(-1).expand_as(coloring_cs)

    return coloring_cs

def cWCT(cont_feat, styl_feat):
        """
        :param cont_feat: [B, N, cH, cW]
        :param styl_feat: [B, N, sH, sW]
        :return color_fea: [B, N, cH, cW]
        """
        B, N, cH, cW = cont_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)
        styl_feat = styl_feat.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        cont_feat = cont_feat.double()
        styl_feat = styl_feat.double()

        # whitening and coloring transforms
        whiten_fea = whitening(cont_feat)
        color_fea = coloring(whiten_fea, styl_feat)

        color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)
###############################################################################
###############################################################################
###############################################################################

def covariance_loss(fc, fs, normalize=False):
    b, c, _, _ = fc.shape

    fc = fc.reshape(b, c, -1)
    fs = fs.reshape(b, c, -1)

    # center
    fc = fc - fc.mean(2, keepdims=True)
    fs = fs - fs.mean(2, keepdims=True)

    if normalize: # 结构保存的很好，风格迁移大多失败
        fc_norm = ((fc * fc).sum(1, keepdims=True) + 1e-8).sqrt()
        fs_norm = ((fs * fs).sum(1, keepdims=True) + 1e-8).sqrt()

        fc = fc / (fc_norm + 1e-8)
        fs = fs / (fs_norm + 1e-8)

    # 计算协方差阵
    cov_c = torch.bmm(fc, fc.transpose(1,2)).div(fc.shape[1])
    cov_s = torch.bmm(fs, fs.transpose(1,2)).div(fs.shape[1])

    return torch.mean((cov_c - cov_s)**2)

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

    # LOGGER.debug(A.shape)
    # LOGGER.debug(B.shape)
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


def calc_ss_loss(A, B):
    # A,B: [BCHW]
    MA = cosine_dismat(A, A)
    MB = cosine_dismat(B, B)
    # Lself_similarity = torch.abs(MA-MB).mean() # Lss=1/HW sum{cosine-similarity-matirx}
    Lself_similarity = torch.mean((MA-MB)**2) # Lss=1/HW sum{cosine-similarity-matirx}
    ##################################测试用###########################################
    # print_gpu_mem()
    # recorder.save_record('peak_mem_alloc', torch.cuda.memory_allocated() / 1024**2)
    # recorder.save_record('dmat_mem_alloc', calc_mem_allocated(MA) * 3)
    ##################################测试用###########################################
    return Lself_similarity


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


def st_target(A, B):
    A = A.reshape(A.shape[0], A.shape[1], -1)
    B = B.reshape(B.shape[0], B.shape[1], -1)

    A = A - A.mean(2, keepdims=True)
    B = B - B.mean(2, keepdims=True)

    # # [B,HW] 计算逐像素点的每个通道像素值组成的向量的模。
    # A_norm = torch.sqrt((A**2).sum(1)) 
    # B_norm = torch.sqrt((B**2).sum(1))

    # # 计算cosine-similarity前标准化
    # A = (A/A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0,2,1) 
    # B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))

    # HW,C mul C,HW 逐点计算(每个像素点位置所有通道的像素值构成一个向量)cos距离
    crossattn = torch.bmm(B, B.transpose(1, 2)) #! 显存占用大户
    selfattn = torch.bmm(A, A.transpose(1, 2))
    fo = torch.bmm(A.transpose(1, 2), selfattn)
    return torch.mean((fo - torch.bmm(A.transpose(1, 2), crossattn)) ** 2)


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


"""
Copyright 2021 Mahmoud Afifi.
 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN: 
 Controlling Colors of GAN-Generated and Real Images via Color Histograms." 
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via 
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
"""

EPS = 1e-6

class RGBuvHistBlock(nn.Module):
  def __init__(self, h=64, insz=150, resizing='interpolation',
               method='inverse-quadratic', sigma=0.02, intensity_scale=True,
               device='cuda'):
    """ Computes the RGB-uv histogram feature of a given image.
    Args:
      h: histogram dimension size (scalar). The default value is 64.
      insz: maximum size of the input image; if it is larger than this size, the 
        image will be resized (scalar). Default value is 150 (i.e., 150 x 150 
        pixels).
      resizing: resizing method if applicable. Options are: 'interpolation' or 
        'sampling'. Default is 'interpolation'.
      method: the method used to count the number of pixels for each bin in the 
        histogram feature. Options are: 'thresholding', 'RBF' (radial basis 
        function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
      sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is 
        the sigma parameter of the kernel function. The default value is 0.02.
      intensity_scale: boolean variable to use the intensity scale (I_y in 
        Equation 2). Default value is True.

    Methods:
      forward: accepts input image and returns its histogram feature. Note that 
        unless the method is 'thresholding', this is a differentiable function 
        and can be easily integrated with the loss function. As mentioned in the
         paper, the 'inverse-quadratic' was found more stable than 'RBF' in our 
         training.
    """
    super(RGBuvHistBlock, self).__init__()
    self.h = h
    self.insz = insz
    self.device = device
    self.resizing = resizing
    self.method = method
    self.intensity_scale = intensity_scale
    if self.method == 'thresholding':
      self.eps = 6.0 / h
    else:
      self.sigma = sigma

  def forward(self, x):
    x = torch.clamp(x, 0, 1)
    if x.shape[2] > self.insz or x.shape[3] > self.insz:
      if self.resizing == 'interpolation':
        x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                  mode='bilinear', align_corners=False)
      elif self.resizing == 'sampling':
        inds_1 = torch.LongTensor(
          np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
          device=self.device)
        inds_2 = torch.LongTensor(
          np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
          device=self.device)
        x_sampled = x.index_select(2, inds_1)
        x_sampled = x_sampled.index_select(3, inds_2)
      else:
        raise Exception(
          f'Wrong resizing method. It should be: interpolation or sampling. '
          f'But the given value is {self.resizing}.')
    else:
      x_sampled = x

    L = x_sampled.shape[0]  # size of mini-batch
    if x_sampled.shape[1] > 3:
      x_sampled = x_sampled[:, :3, :, :]
    X = torch.unbind(x_sampled, dim=0)
    hists = torch.zeros((x_sampled.shape[0], 3, self.h, self.h)).to(
      device=self.device)
    for l in range(L):
      I = torch.t(torch.reshape(X[l], (3, -1)))
      II = torch.pow(I, 2)
      if self.intensity_scale:
        Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS), 
                             dim=1)
      else:
        Iy = 1

      Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] + EPS),
                            dim=1)
      Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] + EPS),
                            dim=1)
      diff_u0 = abs(
        Iu0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      diff_v0 = abs(
        Iv0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      if self.method == 'thresholding':
        diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
        diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u0 = torch.exp(-diff_u0)  # Radial basis function
        diff_v0 = torch.exp(-diff_v0)
      elif self.method == 'inverse-quadratic':
        diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
        diff_v0 = 1 / (1 + diff_v0)
      else:
        raise Exception(
          f'Wrong kernel method. It should be either thresholding, RBF,' 
          f' inverse-quadratic. But the given value is {self.method}.')
      diff_u0 = diff_u0.type(torch.float32)
      diff_v0 = diff_v0.type(torch.float32)
      a = torch.t(Iy * diff_u0)
      hists[l, 0, :, :] = torch.mm(a, diff_v0)

      Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                            dim=1)
      Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                            dim=1)
      diff_u1 = abs(
        Iu1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      diff_v1 = abs(
        Iv1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))

      if self.method == 'thresholding':
        diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
        diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u1 = torch.exp(-diff_u1)  # Gaussian
        diff_v1 = torch.exp(-diff_v1)
      elif self.method == 'inverse-quadratic':
        diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
        diff_v1 = 1 / (1 + diff_v1)

      diff_u1 = diff_u1.type(torch.float32)
      diff_v1 = diff_v1.type(torch.float32)
      a = torch.t(Iy * diff_u1)
      hists[l, 1, :, :] = torch.mm(a, diff_v1)

      Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] + EPS),
                            dim=1)
      Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] + EPS),
                            dim=1)
      diff_u2 = abs(
        Iu2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      diff_v2 = abs(
        Iv2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      if self.method == 'thresholding':
        diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
        diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u2 = torch.exp(-diff_u2)  # Gaussian
        diff_v2 = torch.exp(-diff_v2)
      elif self.method == 'inverse-quadratic':
        diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
        diff_v2 = 1 / (1 + diff_v2)
      diff_u2 = diff_u2.type(torch.float32)
      diff_v2 = diff_v2.type(torch.float32)
      a = torch.t(Iy * diff_u2)
      hists[l, 2, :, :] = torch.mm(a, diff_v2)

    # normalization
    hists_normalized = hists / (
        ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

    return hists_normalized
  
def calc_histogram_loss(A, B, histogram_block):
    input_hist = histogram_block(A) # [4, 3, 256, 256]
    target_hist = histogram_block(B)
    histogram_loss = (1/np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) / 
        input_hist.shape[0])

    return histogram_loss

