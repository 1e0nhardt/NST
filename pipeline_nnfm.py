from collections import OrderedDict
from dataclasses import dataclass
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
from utils import check_folder, get_basename, str2list, images2gif, CONSOLE
import tyro
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline


@dataclass
class NNFMConfig(OptimzeBaseConfig):
    """NNFM Arguments"""

    output_dir: str = "./results/nnfm"
    """output dir"""
    layers: str = "11,13,15"
    """layer indices of style features which should seperate by ','"""


class NNFMPipeline(OptimzeBasePipeline):
    def __init__(self, config: NNFMConfig) -> None:
        super().__init__(config)
        self.config = config
        CONSOLE.print(config)
        check_folder(self.config.output_dir)

    # def add_extra_infos(self):
    #     return ['100x']
    
    def optimize_process(self, content_path, style_path, mask=False):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        # convert("RGB")会让灰度图像也变为3通道
        style_image = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)

        # optimize target
        optimze_images = []
        opt_img = content_image.data.clone()
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
        optimze_images.append(out_img)

        opt_img.requires_grad = True
        optimizer = optim.Adam([opt_img], lr=1e-1)

        # get target features
        _, style_features = self.vgg.forward(style_image)
        _, content_features = self.vgg.forward(content_image)
        s_feats = torch.cat(style_features, 1)
        c_feats = torch.cat(content_features, 1)

        for i in range(self.config.max_iter):
            optimizer.zero_grad()
            _, x_features = self.vgg.forward(opt_img)

            # combine feature channels
            x_feats = torch.cat(x_features, 1)

            # calc losses
            target_feats = feat_replace(x_feats, s_feats) #? 实现是否正确?
            nn_loss = cos_loss(x_feats, target_feats)
            content_loss = torch.mean((c_feats - x_feats) ** 2)
            total_variance_loss = tv_loss(opt_img)
            loss = nn_loss + 1e-4 * content_loss + total_variance_loss

            loss.backward()
            # 将不需要优化的区域的梯度置为零
            if mask:
                opt_img.grad[~self.erase_mask] = 0 
            optimizer.step()

            #print loss
            if i % self.config.show_iter == self.config.show_iter - 1:
                CONSOLE.print(f'Iteration: {i+1}, loss: {loss.item():.4f}, nn_loss: {nn_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                optimze_images.append(out_img)

        # save results
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimze_images.append(out_img)

        return optimze_images


if __name__ == '__main__':
    args = tyro.cli(NNFMConfig)
    pipeline = NNFMPipeline(args)
    pipeline(
        'data/content/sailboat.jpg', 
        'data/style/vangogh_starry_night.jpg',
        mask=False
    )