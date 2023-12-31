import random
from dataclasses import dataclass

import torch
import tyro

from losses import cos_loss
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from image_pyramid import dec_lap_pyr, syn_lap_pyr
from utils import LOGGER, AorB
from utils_st import (calc_remd_loss, feat_replace, nnst_fs_loss,
                            split_downsample)


@dataclass
class NNSTConfig(OptimzeBaseConfig):
    """My reimplement of a simplified version of NNST."""
    
    name: str = 'nnst'
    """expr name. Do not change this."""
    model_type: str = 'vgg16'
    """feature extractor model type vgg16 | vgg19 | inception"""
    # layers: str = "22, 20, 18"
    layers: str = "25, 18, 11, 6, 1"
    # layers: str = "22, 20, 18, 15, 13, 11, 8, 6, 3, 1"
    """layer indices of style features which should seperate by ','"""
    alpha: float = 0.25
    """stylization level """
    max_scales: int = 4
    """do HM/FS on how many image pyramid levels"""
    pyramid_levels: int = 8
    """levels of image pyramid"""
    use_content_loss: bool = False
    """use content loss or not"""
    lr: float = 2e-3
    max_iter: int = 200
    standardize: bool = True
    split_downsample: bool = False
    weight_factor: float = 1.0
    use_remd_loss: bool = True
    use_wandb: bool = False
    exact_resize: bool = True


class NNSTPipeline(OptimzeBasePipeline):
    def __init__(self, config: NNSTConfig) -> None:
        super().__init__(config)
        self.config = config

    def add_extra_file_infos(self):
        return []
        # [str(self.config.alpha)] + [str(self.config.weight_factor)] + AorB(self.config.split_downsample, 'split', 'stride') + AorB(self.config.matting_laplacian_regularizer, f'mlr-{self.config.mlr_weight}')
    
    def add_extra_infos(self):
        return AorB(self.config.use_remd_loss, 'remd')
    
    def common_optimize_process(self, Ic, Is):
        self.writer.log_image('content', self.tensor2pil(Ic))
        self.writer.log_image('reference', self.tensor2pil(Is))

        # 构建图像金字塔
        style_pyramid = dec_lap_pyr(Is, self.config.pyramid_levels)
        content_pyramid = dec_lap_pyr(Ic, self.config.pyramid_levels)
        output_pyramid = dec_lap_pyr(Ic.clone(), self.config.pyramid_levels)

        # LOGGER.debug('=================Pyramid shape=====================')
        # for x in output_pyramid:
        #     LOGGER.debug(x.shape)
        # LOGGER.debug('='*51)

        #! 用低分辨率的内容图初始化
        for i in range(3):
            output_pyramid[i] *= 0
        
        #! Stylize using Hypercolumn Matching from coarse to fine scale
        for scale in range(self.config.max_scales)[::-1]:
            # 合成当前分辨率的内容图，风格图，输出图
            style_image_tmp = syn_lap_pyr(style_pyramid[scale:])
            content_image_tmp = syn_lap_pyr(content_pyramid[scale:])
            output_image_tmp = syn_lap_pyr(output_pyramid[scale:])
            LOGGER.info(f'Stage {self.config.max_scales - scale}: {output_image_tmp.shape}')

            #! 获取目标特征
            with torch.no_grad():
                # alpha用于控制风格化程度
                if scale == self.config.max_scales-1:
                    alpha = 0
                else:
                    alpha = self.config.alpha

                # 用内容的高频特征作为寻找特征的起点
                output_init = syn_lap_pyr([content_pyramid[scale]] + output_pyramid[(scale + 1):])
                # 提取参考图片特征 | 可以使用旋转增强flip_aug，但没必要
                _, style_hypercolumns = self.model(style_image_tmp, hypercolumn=True, weight_factor=self.config.weight_factor, normalize=True)
                # 将内容图和低分辨率下优化过的结果混合
                fuse_content = (1 - alpha) * content_image_tmp + alpha * output_init
                # 提混合内容特征
                _, content_hypercolumns = self.model(fuse_content, hypercolumn=True, weight_factor=self.config.weight_factor, normalize=True)
                # 使用参考图的特征替换最近邻内容特征，得到优化的目标特征
                target_feats = feat_replace(content_hypercolumns, style_hypercolumns)

                # 日志
                LOGGER.info(f'Target Features: {target_feats.shape}')

            with torch.no_grad(): # 保存初始图像
                if scale == self.config.max_scales - 1 and False:
                    output_image_tensor = syn_lap_pyr(output_pyramid)
                    self.save_image(output_image_tensor, f'HM_init', verbose=True)
                    self.writer.log_image(f'HM_init', self.tensor2pil(output_image_tensor))

            #! 在当前分辨率下用Hypercolumn Matching优化输出图像
            self.optimize_output(output_pyramid, target_feats, scale)

            with torch.no_grad(): # 保存中间优化结果
                if scale == 0:
                    output_image_tensor = syn_lap_pyr(output_pyramid)
                    self.save_image(output_image_tensor, f'HM_{scale}', verbose=True)
                    self.writer.log_image(f'HM_{scale}', self.tensor2pil(output_image_tensor))

        #! 在最高分辨率下用Feature Split优化输出图像
        LOGGER.info(f'Final Stage: FS')
        self.optimize_output(output_pyramid, None, scale, Is, final_pass=True)

        with torch.no_grad(): # 保存最终输出
            output_image_tensor = syn_lap_pyr(output_pyramid)
            self.save_image(output_image_tensor, 'result')
            self.writer.log_image('result', self.tensor2pil(output_image_tensor))
    
    def optimize_output(self, output_pyramid, target_feats, scale, style_image=None, final_pass=False):
        # 将当前尺度下的图像金字塔设为优化对象
        opt_vars = [torch.nn.Parameter(level_image) for level_image in output_pyramid[scale:]]
        optimizer = torch.optim.Adam(opt_vars, lr=self.config.lr)

        if final_pass: # FS需要重新提取各个尺度的特征
            assert style_image is not None, 'You should pass style image in final pass'
            _, style_features = self.model(style_image, weight_factor=self.config.weight_factor, normalize=True)

        #! 优化过程
        # iter_num = self.config.max_iter if not final_pass else 500
        for i in range(self.config.max_iter):
            optimizer.zero_grad()
            output_image = syn_lap_pyr(opt_vars)

            loss = 0

            if not final_pass:
                # HM
                _, cur_feats = self.model(output_image, hypercolumn=True, weight_factor=self.config.weight_factor, normalize=True)
                # 计算损失
                loss += cos_loss(target_feats, cur_feats)
                # 日志
                self.writer.log_scaler(f'loss at scale {scale}', loss.item())
            else:
                # FS
                _, cur_feats = self.model(output_image, weight_factor=self.config.weight_factor, normalize=True)
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
                    if self.config.use_remd_loss:
                        loss += calc_remd_loss(style, cur)
                    else:
                        loss += nnst_fs_loss(style, cur)

                # 日志
                # LOGGER.warning(f'loss: {loss.item()}')
                self.writer.log_scaler(f'loss of FS', loss.item())

            loss.backward()
            optimizer.step()
            
        # if not final_pass:
            # self.writer.add_embedding(self.feat2embedding(cur_feats), global_step=self.config.max_scales-1-scale, tag='curr_optimzed')
            # self.writer.add_histogram('cur_opt', self.feat2embedding(cur_feats).mean(1))
        
        # 用优化好的图像重新构建图像金字塔，替换原输出图像金字塔的对应部分
        output_pyramid[scale:] = dec_lap_pyr(output_image, self.config.pyramid_levels - scale)

if __name__ == '__main__':
    import os
    args = tyro.cli(NNSTConfig)
    pipeline = NNSTPipeline(args)
    content_dir='data/contents/'
    style_dir='data/styles/'
    for c in os.listdir(content_dir):
        for s in os.listdir(style_dir):
            pipeline(
                content_dir + c,
                style_dir + s
            )
    # exit()
    # for s in os.listdir(style_dir):
    #     pipeline(
    #         'data/content/sailboat.jpg', 
    #         style_dir + s
    #     )
        
    # pipeline(
    #     # 'data/truck/14.png', 
    #     'data/content/sailboat.jpg', 
    #     # 'data/content/C2.png', 
    #     # 'data/style/122.jpg',
    #     'data/nnst_style/S1.jpg'
    # )

    # pipeline(
    #     'data/content/content_im.jpg', 
    #     'data/content/style_im.jpg',
    # )
