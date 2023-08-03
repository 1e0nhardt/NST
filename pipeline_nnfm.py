from collections import OrderedDict
from dataclasses import dataclass
import random
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from PIL import Image
from torch import optim
from torch.autograd import Variable
from torchvision import models, transforms
from torchvision.utils import save_image


from losses import gram_loss, tv_loss, feat_replace, cos_loss
from models.vgg import VGG16FeatureExtractor, VGG19FeatureExtractor
from utils import AorB, ResizeHelper, check_folder, get_basename, str2list, images2gif, LOGGER
import tyro
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from utils_st import calc_remd_loss, nnst_fs_loss, split_downsample


@dataclass
class NNFMConfig(OptimzeBaseConfig):
    """NNFM Arguments"""

    output_dir: str = "./results/nnfm"
    """output dir"""
    model_type: str = 'vgg16'
    # layers: str = "11,13,15"
    layers: str = "25, 18, 11, 6, 1"
    """layer indices of style features which should seperate by ','"""
    lr: float = 1e-2
    max_iter: int = 500
    method: str = 'FS'
    use_remd_loss: bool = True
    weight_factor: float = 1
    split_downsample: bool = False
    use_wandb: bool = False


class NNFMPipeline(OptimzeBasePipeline):
    def __init__(self, config: NNFMConfig) -> None:
        super().__init__(config)
        self.config = config
        check_folder(self.config.output_dir)
        self.kl_loss = nn.KLDivLoss()

    def add_extra_file_infos(self):
        return [str(self.config.weight_factor)] + AorB(self.config.split_downsample, 'split', 'stride') + [str(self.config.method)]
    
    def add_extra_infos(self):
        return AorB(self.config.use_remd_loss, 'remd') + ['cinit_tvloss_l2']
    
    def optimize_process(self, content_path, style_path, mask=False):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        # convert("RGB")会让灰度图像也变为3通道
        style_image = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)
        resize_helper = ResizeHelper()
        content_image = resize_helper.resize_to8x(content_image)
        style_image = resize_helper.resize_to8x(style_image)

        # optimize target
        optimize_images = OrderedDict()
        opt_img = content_image.data.clone()
        alpha = 0.0
        opt_img = (1-alpha)*opt_img + alpha*style_image.data.clone()
        # opt_img[:,:,-150:,-150:] = (opt_img[:,:,-150:,-150:] + style_image.data.clone()[:,:,-150:,-150:])/2
        # opt_img = torch.rand_like(content_image)
        opt_img = opt_img.to(self.device)
        if mask:
            opt_img = self.transform_pre(Image.open(
                self.generate_expr_name()+f"/{self.generate_filename(content_path, style_path)}_result.png"
            )).unsqueeze(0).to(self.device)

            # generate mask
            # erase_mask = torch.zeros_like(opt_img[0, 0])
            # erase_mask[0:300, 0:300] = 1
            # erase_mask = (erase_mask == 1)
            # erase_mask = erase_mask[None, None, ...]
            # self.erase_mask = repeat(erase_mask, 'b c h w -> b (r c) h w', r=3)
            self.erase_mask = torch.rand_like(opt_img) > 0.4

            opt_img[self.erase_mask] = content_image[self.erase_mask]
            
        # save init image
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images['init'] = out_img

        opt_img.requires_grad = True
        optimizer = optim.Adam([opt_img], lr=self.config.lr)

        # get target features
        _, style_features = self.model.forward(style_image, hypercolumn=self.config.method == 'HM')
        _, content_features = self.model.forward(content_image, hypercolumn=self.config.method == 'HM')
        s_feats = style_features
        c_feats = content_features
        # s_feats = torch.cat(style_features, 1)
        # c_feats = torch.cat(content_features, 1)
        if self.config.method == 'HM':
            target_feats = feat_replace(c_feats, s_feats)

        for i in range(self.config.max_iter):
            optimizer.zero_grad()

            loss = 0
            if self.config.method == 'HM':
                _, x_features = self.model.forward(opt_img, hypercolumn=True)

                # combine feature channels
                x_feats = x_features
                # x_feats = torch.cat(x_features, 1)

                # calc losses
                nn_loss = cos_loss(x_feats, target_feats)
                # content_loss = torch.mean((c_feats - x_feats) ** 2)
                # total_variance_loss = tv_loss(opt_img)
                loss = nn_loss# + 1e-4 * content_loss + total_variance_loss
                (x_feats, target_feats)
                # loss += 1e-3 * self.kl_loss(x_feats, target_feats)
            elif self.config.method == 'FS':
                _, cur_feats = self.model(opt_img, weight_factor=self.config.weight_factor)
                    # 每一层的特征分别找最近邻并计算cosine距离
                for cur, style in zip(cur_feats, style_features):
                    # 如果特征图宽高太大，则进行稀疏采样
                    if max(cur.size(2), cur.size(3)) > 64:
                        stride = max(cur.size(2), cur.size(3)) // 64
                        if self.config.split_downsample:
                            style_lst = split_downsample(style, stride)
                            cur_lst = split_downsample(cur, stride)
                            style = torch.cat(style_lst, dim=1)
                            cur = torch.cat(cur_lst, dim=1)
                        else:
                            offset_a = random.randint(0, stride - 1)
                            offset_b = random.randint(0, stride - 1)
                            style = style[:, :, offset_a::stride, offset_b::stride]
                            cur = cur[:, :, offset_a::stride, offset_b::stride]
                    if i == 0:
                        LOGGER.debug(f'Shape of features: {style.shape} {cur.shape}')
                    # 计算损失
                    # if self.config.use_remd_loss:
                    #     loss += calc_remd_loss(style, cur)
                    # else:
                    #     loss += nnst_fs_loss(style, cur)
                    for c, s in zip(torch.split(cur, 1, dim=1), torch.split(style, 1, dim=1)):
                        if self.config.use_remd_loss:
                            loss += calc_remd_loss(s, c, l2=True)
                        else:
                            loss += nnst_fs_loss(s, c)
                    
                    # loss += 1e1 * (style * (torch.log(style) - torch.log(cur))).mean()

            total_variance_loss = tv_loss(opt_img)
            loss += 10 * total_variance_loss
            # 日志
            self.writer.log_scaler(f'loss of {self.config.method}', loss.item())

            loss.backward()
            self.writer.log_scaler('loss', loss.item())
            # 将不需要优化的区域的梯度置为零
            if mask:
                opt_img.grad[~self.erase_mask] = 0 
            optimizer.step()

            #print loss
            if i % self.config.show_iter == self.config.show_iter - 1:
                LOGGER.info(f'Iteration: {i+1}, loss: {loss.item():.4f}, tv_loss: {total_variance_loss.item():.1e}')
                # LOGGER.info(f'Iteration: {i+1}, loss: {loss.item():.4f}, nn_loss: {nn_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                optimize_images[f'iter_{i}'] = out_img

        # save results
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images['result'] = out_img
        self.writer.log_image('result', out_img)

        return optimize_images


if __name__ == '__main__':
    import os
    args = tyro.cli(NNFMConfig)
    pipeline = NNFMPipeline(args)
    style_dir='data/nnst_style/'
    # for s in os.listdir(style_dir):
    #     pipeline(
    #         'data/content/sailboat.jpg',
    #         style_dir + s
    #     )

    pipeline(
        # 'data/content/C1.png', 
        'data/content/sailboat.jpg', 
        'data/nnst_style/shape.png',
        # 'data/style/17.jpg',
    )