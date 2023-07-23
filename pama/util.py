import torch
import torch.nn as nn
import numpy as np
from utils import recorder, print_gpu_mem, calc_mem_allocated
from torch.autograd import Function

class RemdLoss(Function):
    """
    M = A @ B \n
    remd = max(mean(M.min(1)), mean(M.min(2))) \n
    输入必须标准化过，且形状如下 \n
    A: (1, hw, c) \n
    B: (1, c, hw)
    """

    @staticmethod
    def forward(ctx, A, B, step=100):
        b, hw, c = A.shape
        device = A.device
        assert b == 1, f"batch size should be 1, however {b} is accepted"
        row_min = torch.empty([hw], device=device)
        row_min_index = torch.arange(hw, device=device).unsqueeze(1).repeat([1, 2])
        col_min = torch.ones([hw], device=device)
        col_min_index = torch.arange(hw, device=device).unsqueeze(1).repeat([1, 2])

        #################################逐行进行#######################################
        # temp_row = torch.empty([hw], device=device)
        # for i in range(hw):
        #     temp_row = (1 - torch.bmm(A[:, i:i+1], B)).squeeze() #! 引入了一个负号
        #     #! 记录Distance Matrix每一行的最小值及其位置
        #     row_min[i], row_min_index[i, 1] = temp_row.min(dim=-1)
        #     #! 记录Distance Matrix每一列的最小值及其位置
        #     col_min, updated_inds = torch.stack([col_min, temp_row]).min(dim=0) # 始终用当前值和每一列记录的最小值比，取更小的。如果当前值最小，则返回的索引值为1，转换为布尔索引则可以更新所有需要更新的坐标
        #     col_min_index[updated_inds.bool(), 0] = i # 使用布尔索引更新坐标索引
        ################################################################################

        # 批量进行
        for i in range(0, hw, step):
            if i + step > hw:
                real_step = hw - i
            else:
                real_step = step

            temp_rows = (1 - torch.bmm(A[:, i:i+real_step], B)).squeeze(0) #! 引入了一个负号
            if i == 0:
                recorder.save_record('dmat_mem_alloc', calc_mem_allocated(temp_rows))
                
            #! 记录Distance Matrix每一行的最小值及其位置
            row_min[i:i+real_step], min_inds = temp_rows.min(dim=-1) # (step,)
            row_min_index[i:i+real_step, 1] = min_inds
            #! 记录Distance Matrix每一列的最小值及其位置
            col_min, min_inds = torch.cat([temp_rows, col_min.unsqueeze(0)], dim=0).min(dim=0)
            # 使用布尔索引更新坐标索引
            col_min_index[:, 0][min_inds != real_step] = i + min_inds[min_inds != real_step] 

        r = row_min.mean()
        c = col_min.mean()
        remd = max(r, c)

        #####################################测试用########################################
        # print_gpu_mem()
        # recorder.save_record('peak_mem_alloc', torch.cuda.memory_allocated() / 1024**2)
        ###################################################################################

        if r < c:       
            ctx.save_for_backward(col_min_index, A, B)
        else:
            ctx.save_for_backward(row_min_index, A, B)
        return remd

    @staticmethod
    def backward(ctx, grad_output):
        mins_index, A, B = ctx.saved_tensors
        coefficient = grad_output / B.shape[2] * -1
        grad_A = torch.zeros_like(A, device=A.device) # (1, hw, c)
        grad_B = torch.zeros_like(B, device=B.device) # (1, c, hw)
        #! 串行导致计算速度慢。可能并不是。
        #! 可能是矩阵元素远大于cuda可以并行的线程数量，导致需要GPU执行的波数变多，此外访存操作也同样变多。性能瓶颈可能在这。
        #! 实际就是矩阵规模O(hw^2)增长的问题。
        for i, j in mins_index:
            grad_A[0, i] += B[0, :, j]
            grad_B[0, :, j] += A[0, i]   
        return coefficient * grad_A, coefficient * grad_B, None


class SelfSimilarityLoss(Function):
    """
    MA = 1 - AT @ A, MB = 1 - BT @ B \n
    ss = mean(abs(MA - MB)) \n
    输入必须标准化过，且形状如下 \n
    AT, BT: (1, hw, c) \n
    A, BT: (1, c, hw) \n
    关于forward添加非tensor参数的问题 https://github.com/pytorch/pytorch/issues/16940
    """

    @staticmethod
    def forward(ctx, AT, A, BT, B, step:int = 100):
        b, hw, c = AT.shape
        assert b == 1, f"batch size should be 1, however {b} is accepted"
        device = A.device

        grad_AT = torch.empty([hw, c], device=device) # (1, hw, c)
        grad_A = torch.empty([c, hw], device=device) # (1, c, hw)
        grad_BT = torch.empty([hw, c], device=device) # (1, hw, c)
        grad_B = torch.empty([c, hw], device=device) # (1, c, hw)

        dis_mean = torch.empty([hw], device=device)
        # step越大，显存占用也会变大。速度会先升再降。
        for i in range(0, hw, step):
            # 处理最后一批可能不足step的情况
            if i + step > hw:
                real_step = hw - i
            else:
                real_step = step
            # forward
            MA_rows = (1 - torch.bmm(AT[:, i:i+real_step], A)).squeeze(0) #! 引入了一个负号
            MB_rows = (1 - torch.bmm(BT[:, i:i+real_step], B)).squeeze(0) #! 引入了一个负号
            if i == 0:
                recorder.save_record('dmat_mem_alloc', calc_mem_allocated(MA_rows) * (4 + c))

            D_rows = MA_rows - MB_rows
            dis_mean[i:i+real_step] = torch.abs(D_rows).mean(dim=-1) #! abs结果为0时，梯度直接设为0

            # compute grad for backward proces
            D_row_sign = D_rows.clone()
            D_row_sign[D_rows > 0] = 1
            D_row_sign[D_rows < 0] = -1

            #####################################逐行处理######################################
            # grad_AT[i, :] = (A * D_row_sign).squeeze().sum(-1)
            # grad_BT[i, :] = (B * D_row_sign * -1).squeeze().sum(-1)
            # grad_A += AT[0, i, :].unsqueeze(1).repeat(1, hw) * D_row_sign
            # grad_B += BT[0, i, :].unsqueeze(1).repeat(1, hw) * D_row_sign * -1
            ###################################################################################

            # 批量处理写法。注意尾端特殊情况
            grad_AT[i:i+real_step, :] = \
                (A.repeat(1, real_step, 1) * D_row_sign.repeat_interleave(c, dim=0)).squeeze().sum(dim=-1).reshape(real_step, c)
            grad_BT[i:i+real_step, :] = \
                (B.repeat(1, real_step, 1) * D_row_sign.repeat_interleave(c, dim=0) * -1).squeeze().sum(dim=-1).reshape(real_step, c)

            grad_A += (AT[0, i:i+real_step, :].unsqueeze(2).repeat(1, 1, hw) * D_row_sign.unsqueeze(1)).sum(dim=0)
            grad_B += (BT[0, i:i+real_step, :].unsqueeze(2).repeat(1, 1, hw) * D_row_sign.unsqueeze(1) * -1).sum(dim=0)
            
        ss = dis_mean.mean()
        ##################################测试用###########################################
        # recorder.save_record('peak_mem_alloc', torch.cuda.memory_allocated() / 1024**2)
        # print_gpu_mem()
        ###################################################################################
        ctx.save_for_backward(grad_AT, grad_A, grad_BT, grad_B)
        return ss

    @staticmethod
    def backward(ctx, grad_output):
        grad_AT, grad_A, grad_BT, grad_B = ctx.saved_tensors
        _, hw = grad_A.shape
        coefficient = grad_output / (hw ** 2) * -1
        return (
            coefficient * grad_AT.unsqueeze(0),
            coefficient * grad_A.unsqueeze(0),
            coefficient * grad_BT.unsqueeze(0),
            coefficient * grad_B.unsqueeze(0),
            None
        )


# B, C, H, W; mean var on HW
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def calc_remd_loss_custom(A, B, step=64):
    """
    使用自定义算子计算REMD损失，需要的显存从O(hw^2)降为O(hw)，但因为计算从并行变为串行，计算速度会变慢。 \n
    A, B: (1, c, h, w) \n
    M = A @ B \n
    remd = max(mean(M.min(1)), mean(M.min(2)))
    """
    b, c, h, w = A.shape

    if h * w < 129**2:
        return calc_remd_loss(A, B)

    A = A.view(b, c, h*w)
    B = B.view(b, c, h*w)

    A_norm = torch.sqrt((A**2).sum(1)) # [B,HW] 计算逐像素点的每个通道像素值组成的向量的模。
    B_norm = torch.sqrt((B**2).sum(1))

    A = (A/A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0,2,1) # 计算cosine-similarity前标准化
    B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))

    remd = RemdLoss.apply(A, B, step)
    return remd


def calc_ss_loss_custom(A, B, step=64):
    """
    使用自定义算子计算自相似损失，需要的显存从O(hw^2)降为O(hw)，但因为计算从并行变为串行，计算速度会变慢。 \n
    A, B: (1, c, h, w) \n
    MA = 1 - AT @ A, MB = 1 - BT @ B \n
    ss = mean(abs(MA - MB)) \n
    A,B的形状必须完全相同。
    """
    b, c, h, w = A.shape

    if h * w < 129**2:
        return calc_ss_loss(A, B)

    A = A.view(b, c, -1)
    B = B.view(b, c, -1)

    A_norm = torch.sqrt((A**2).sum(1)) # [B,HW] 计算逐像素点的每个通道像素值组成的向量的模。
    B_norm = torch.sqrt((B**2).sum(1))

    # 计算cosine-similarity前标准化
    A = (A/A_norm.unsqueeze(dim=1).expand(A.shape))
    AT = A.permute(0, 2, 1) 
    B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))
    BT = B.permute(0, 2, 1)

    ss = SelfSimilarityLoss.apply(AT, A, BT, B, step)
    return ss


def cosine_dismat(A, B):
    A = A.view(A.shape[0], A.shape[1], -1)
    B = B.view(B.shape[0], B.shape[1], -1)

    A_norm = torch.sqrt((A**2).sum(1)) # [B,HW] 计算逐像素点的每个通道像素值组成的向量的模。
    B_norm = torch.sqrt((B**2).sum(1))

    A = (A/A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0,2,1) # 计算cosine-similarity前标准化
    B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))
    # HW,C mul C,HW 逐点计算(每个像素点位置所有通道的像素值构成一个向量)cos距离
    dismat = 1.-torch.bmm(A, B) #! 显存占用大户

    return dismat


def calc_remd_loss(A, B):
    # A,B: [BCHW]
    C = cosine_dismat(A, B)
    m1, _ = C.min(1) # cosine距离矩阵每列的最小值
    m2, _ = C.min(2) # cosine距离矩阵每行的最小值
    remd = torch.max(m1.mean(), m2.mean())
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
    Lself_similarity = torch.abs(MA-MB).mean() # Lss=1/HW sum{cosine-similarity-matirx}
    ##################################测试用###########################################
    # print_gpu_mem()
    # recorder.save_record('peak_mem_alloc', torch.cuda.memory_allocated() / 1024**2)
    # recorder.save_record('dmat_mem_alloc', calc_mem_allocated(MA) * 3)
    ##################################测试用###########################################
    return Lself_similarity


def calc_moment_loss(A, B):
    A = A.view(A.shape[0], A.shape[1], -1) # [4, 256, 4096]
    B = B.view(B.shape[0], B.shape[1], -1)

    mu_a = torch.mean(A, 1, keepdim=True) # 逐点均值 [4, 1, 4096]
    mu_b = torch.mean(B, 1, keepdim=True)
    mu_d = torch.abs(mu_a - mu_b).mean()

    A_c = A - mu_a
    B_c = B - mu_b
    cov_a = torch.bmm(A_c, A_c.permute(0,2,1)) / (A.shape[2]-1) # [4, 256, 256]
    cov_a = torch.nan_to_num(cov_a) #! 处理inf, nan
    cov_b = torch.bmm(B_c, B_c.permute(0,2,1)) / (B.shape[2]-1)
    cov_b = torch.nan_to_num(cov_b)
    cov_d = torch.abs(cov_a - cov_b).mean()
    loss = mu_d + cov_d
    return loss


def calc_mse_loss(A, B):
    return nn.MSELoss(A, B)


if __name__ == '__main__':
    from torchvision import models
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval()
    print(vgg)
    x = torch.rand([1, 3, 512, 512])
    print(vgg.features[:4](x).shape)
    print(vgg.features[:9](x).shape)
    print(vgg.features[:16](x).shape)
    print(vgg.features[:23](x).shape)
    print(vgg.features[:30](x).shape)