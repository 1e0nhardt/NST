from dataclasses import dataclass

import torch.nn as nn
import tyro
from torch import optim

from losses import cos_loss, feat_replace, gram_loss, tv_loss
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from utils import LOGGER, AorB
from utils_st import calc_remd_loss, covariance_loss, nnst_fs_loss


@dataclass
class NNFMConfig(OptimzeBaseConfig):
    """Explore NNFM apporach."""

    name: str = 'nnfm'
    model_type: str = 'vgg16'
    # layers: str = "11,13,15"
    layers: str = "25, 18, 11, 6, 1"
    """layer indices of style features which should seperate by ','"""
    lr: float = 1e-2
    max_iter: int = 300
    method: str = 'FS'
    use_remd_loss: bool = True
    weight_factor: float = 1
    split_downsample: bool = False
    save_init: bool = False
    exact_resize: bool = True


class NNFMPipeline(OptimzeBasePipeline):
    def __init__(self, config: NNFMConfig) -> None:
        super().__init__(config)
        self.config = config

    def add_extra_file_infos(self):
        return []
    
    def add_extra_infos(self):
        return [str(self.config.method)] + AorB(self.config.use_remd_loss, 'remd', 'nn')# + ["wcont_alpha1000"]
    
    def common_optimize_process(self, Ic, Is):
        # init
        opt_img = Ic.data.clone()
        opt_img = opt_img.to(self.device)

        # save init image
        self.save_image(opt_img, 'init', verbose=True)

        opt_img.requires_grad = True
        optimizer = optim.Adam([opt_img], lr=self.config.lr)

        # get target features
        _, Is_features = self.model(Is, hypercolumn=self.config.method == 'HM', normalize=False)
        Ic_41, Ic_features = self.model(Ic, hypercolumn=self.config.method == 'HM', normalize=False)

        if self.config.method == 'HM':
            target_feats = feat_replace(Ic_features, Is_features)

        for i in range(self.config.max_iter):
            loss = 0
            optimizer.zero_grad()

            if self.config.method == 'HM':
                Io_41, x_features = self.model(opt_img, hypercolumn=True, normalize=False)

                # calc losses
                if self.config.use_remd_loss:
                    nn_loss = calc_remd_loss(x_features, Is_features)
                else:
                    nn_loss = nnst_fs_loss(x_features, Is_features)
                loss = nn_loss

            elif self.config.method == 'FS':
                Io_41, Io_features = self.model(opt_img, weight_factor=self.config.weight_factor, normalize=False)

                for cur, style in zip(Io_features, Is_features):
                    cur = self.downsample_features_if_needed(cur, self.config.split_downsample)
                    style = self.downsample_features_if_needed(style, self.config.split_downsample)

                    if i == 0:
                        LOGGER.debug(f'Shape of features: {style.shape} {cur.shape}')

                    # 计算损失
                    if self.config.use_remd_loss:
                        loss += calc_remd_loss(style, cur)# * 1000
                    else:
                        loss += nnst_fs_loss(style, cur)# * 1000

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

    # content_dir='data/contents/'
    # style_dir='data/styles/'
    # for c in os.listdir(content_dir):
    #     for s in os.listdir(style_dir):
    #         pipeline(
    #             content_dir + c,
    #             style_dir + s
    #         )
    # exit()
    
    style_dir='data/nnst_style/'
    # for s in os.listdir(style_dir):
    #     pipeline(
    #         'data/content/sailboat.jpg',
    #         style_dir + s
    #     )
    # exit()

    pipeline(
        # 'data/content/C1.png', 
        'data/content/sailboat.jpg', 
        # 'results/gatys/pyramid_1-6-11-18-25_cov/sailboat_water_pyr_3.png', 
        # 'data/nnst_style/shape.png',
        'data/nnst_style/17.jpg',
        # 'data/style/19.jpg',
    )

    # pipeline(
    #     'data/content/content_im.jpg', 
    #     'data/content/style_im.jpg',
    # )