import random
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import tyro
from PIL import Image
from torch import optim

from losses import cos_loss, feat_replace, gram_loss, tv_loss
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from utils import LOGGER, AorB, ResizeHelper, ResizeMaxSide, check_folder
from utils_st import calc_remd_loss, get_seg_dicts, load_segment, nnst_fs_loss, split_downsample


@dataclass
class NNFMConfig(OptimzeBaseConfig):
    """NNFM Arguments"""

    name: str = 'nnfm'
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
    save_init: bool = True
    exact_resize: bool = True


class NNFMPipeline(OptimzeBasePipeline):
    def __init__(self, config: NNFMConfig) -> None:
        super().__init__(config)
        self.kl_loss = nn.KLDivLoss()

    def add_extra_file_infos(self):
        return [str(self.config.weight_factor)] + AorB(self.config.split_downsample, 'split', 'stride') + [str(self.config.method)]
    
    def add_extra_infos(self):
        return AorB(self.config.use_remd_loss, 'remd') + ['noise']# + ['mask']
    
    def common_optimize_process(self, Ic, Is):
        # init
        # opt_img = Ic.data.clone()
        # opt_img = self.lerp(opt_img, Is.clone(), 0.1)
        # opt_img = (1-alpha)*opt_img + alpha*Is.data.clone()
        # opt_img[:,:,-150:,-150:] = (opt_img[:,:,-150:,-150:] + Is.data.clone()[:,:,-150:,-150:])/2
        # opt_img = torch.rand_like(Ic)
        content_laplacian = self.get_lap_filtered_image(self.content_path)
        LOGGER.debug(content_laplacian.shape)
        # opt_img = content_laplacian + torch.rand_like(content_laplacian) * 0.2
        # opt_img = self.lerp(content_laplacian, torch.rand_like(Ic), 0)
        opt_img = self.lerp(torch.ones_like(Ic), torch.rand_like(Ic), 1)

        opt_img = opt_img.to(self.device)

        # save init image
        self.save_image(opt_img, 'init')

        opt_img.requires_grad = True
        optimizer = optim.Adam([opt_img], lr=self.config.lr)

        # get target features
        _, Is_features = self.model(Is, hypercolumn=self.config.method == 'HM', normalize=False)
        _, Ic_features = self.model(Ic, hypercolumn=self.config.method == 'HM', normalize=False)

        #! mask
        # style_mask = load_segment('data/content/style_guidance.jpg')
        # content_mask = load_segment('data/content/content_guidance.jpg')

        # c_indices_dict, s_feats_dict = get_seg_dicts(Ic_features, Is_features, content_mask, style_mask)

        if self.config.method == 'HM':
            target_feats = feat_replace(Ic_features, Is_features)

        for i in range(self.config.max_iter):
            loss = 0
            optimizer.zero_grad()

            if self.config.method == 'HM':
                _, x_features = self.model(opt_img, hypercolumn=True, normalize=False)

                # calc losses
                nn_loss = cos_loss(x_features, target_feats)
                # content_loss = torch.mean((c_feats - x_feats) ** 2)
                # total_variance_loss = tv_loss(opt_img)
                loss = nn_loss# + 1e-4 * content_loss + total_variance_loss
                # loss += 1e-3 * self.kl_loss(x_feats, target_feats)

            elif self.config.method == 'FS':
                _, Io_features = self.model(opt_img, weight_factor=self.config.weight_factor, normalize=False)

                # loss += gram_loss(Io_features, Is_features)

                ################################################
                ############### Try use Mask ###################
                ################################################

                # # 如果特征图宽高太大，则进行稀疏采样
                # for i in range(len(Io_features)):
                #     if max(Io_features[i].size(2), Io_features[i].size(3)) > 128:
                #         stride = max(Io_features[i].size(2), Io_features[i].size(3)) // 128
                #         Io_features[i] = Io_features[i][:, :, 0::stride, 0::stride]
                #         Io_features[i] = Io_features[i].reshape(Io_features[i].size(0), Io_features[i].size(1), -1)[0] # (c, hw)
                #     else:
                #         Io_features[i] = Io_features[i].reshape(Io_features[i].size(0), Io_features[i].size(1), -1)[0]
                    
                #     #! center output features
                #     Io_features[i] = Io_features[i] - Io_features[i].mean(1, keepdims=True)

                # # 每个label的每一层的特征分别找最近邻并计算cosine距离
                # for label in c_indices_dict.keys():
                #     for i, (cur, masked_style_feats) in enumerate(zip(Io_features, s_feats_dict[label])):
                #         masked_content_feats = torch.index_select(cur, 1, c_indices_dict[label][i])
                #         if self.config.use_remd_loss:
                #             loss += calc_remd_loss(masked_style_feats.unsqueeze(0), masked_content_feats.unsqueeze(0), center=False)
                #         else:
                #             loss += nnst_fs_loss(masked_style_feats.unsqueeze(0), masked_content_feats.unsqueeze(0), center=False)
                
                ################################################
                ############# Try use Mask  End ################
                ################################################

                for cur, style in zip(Io_features, Is_features):
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
                    if self.config.use_remd_loss:
                        loss += calc_remd_loss(style, cur)
                    else:
                        loss += nnst_fs_loss(style, cur)

                    # 不可行，因为如果分开计算，则每个滤波器都需要保存一个距离矩阵，因为梯度下降需要。但这样对显存大小要求极高。
                    # for c, s in zip(torch.split(cur, 1, dim=1), torch.split(style, 1, dim=1)):
                    #     if self.config.use_remd_loss:
                    #         loss += calc_remd_loss(s, c, l2=True)
                    #     else:
                    #         loss += nnst_fs_loss(s, c)

            # total_variance_loss = tv_loss(opt_img)
            # loss += 10 * total_variance_loss
            loss.backward()
            optimizer.step()

            # 日志
            self.writer.log_scaler(f'loss of {self.config.method}', loss.item())

            #print loss
            if i % self.config.show_iter == self.config.show_iter - 1:
                LOGGER.info(f'Iteration: {i+1}, loss: {loss.item():.4f}')
                # LOGGER.info(f'Iteration: {i+1}, loss: {loss.item():.4f}, tv_loss: {total_variance_loss.item():.1e}')
                self.save_image(opt_img, f'iter_{i}', verbose=True)

        # save results
        self.save_image(opt_img, 'result')
        self.writer.log_image('result', self.tensor2pil(opt_img))


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
        # 'data/nnst_style/shape.png',
        'data/style/6.jpg',
    )

    # pipeline(
    #     'data/content/content_im.jpg', 
    #     'data/content/style_im.jpg',
    # )