from dataclasses import dataclass

import torch
import torch.nn.functional as F
import tyro
from torch import optim

from losses import gram_loss, tv_loss
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from image_pyramid import dec_lap_pyr, syn_lap_pyr
from utils import LOGGER, AorB
from utils_st import (calc_mean_std, calc_remd_loss, calc_ss_loss,
                            covariance_loss, nnst_fs_loss)


@dataclass
class GatysConfig(OptimzeBaseConfig):
    """Gatys NST Arguments"""

    name: str = 'gatys'
    max_iter: int = 300
    optimizer: str = 'lbfgs'
    """optimizer type: adam(x) | lbfgs"""
    optimize_strategy: str = 'pyramid'
    exact_resize: bool = True
    # layers: str = "1"
    layers: str = "1,6,11,18,25"
    """layer indices of style features which should seperate by ','"""
    verbose: bool = False
    content_path: str = 'data/contents/5.jpg'
    style_path: str = 'data/styles/12.jpg'

    max_scales: int = 2
    """do ST on how many image pyramid levels"""
    pyramid_levels: int = 8
    """levels of image pyramid"""
    # matting_laplacian_regularizer: bool = False
    # mlr_weight: float = 1e8
    input_range: float = 1.0
    method: str = 'ours'
    """gatys, gatysdivc, mean, std, adain, cov, l2, efdm, ss, remd, ours"""
    use_tensorboard: bool = False
    visualize_grad: bool = False
    visualize_features: bool = False


class GatysPipeline(OptimzeBasePipeline):
    def __init__(self, config: GatysConfig) -> None:
        super().__init__(config)
        self.config = config # 让编辑器可以准确识别self.config的类型
    
    def add_extra_infos(self):
        return AorB(self.config.method == 'common', self.config.method, self.config.method + '_' + str(self.config.max_scales))# + AorB(self.config.add_weights, 'weights') #+ AorB(self.config.matting_laplacian_regularizer, f'MLR_{self.config.mlr_weight}')

    def check_method(self, i):
        if i in self.config.method:
            return True
        elif i in "gatys, gatysdivc, mean, std, adain, cov, l2, efdm, ss, remd, ours":
            return False
        else:
            raise NotImplementedError(f'This method {self.config.method} is not implemented yet!')
    
    def common_optimize_process(self, Ic, Is):
        opt_img = Ic.data.clone()
        opt_img = opt_img.to(self.device)

        # save init image
        self.save_image(opt_img, 'init', verbose=True)

        # get target features
        _, Is_features = self.model(Is)
        Ic_41, Ic_features = self.model(Ic)

        #! load channel sort indices
        # all_layer_inds = torch.load('data/sailboat_channel_indices.pt')
        all_layer_weights = []

        for fc, fs in zip(Ic_features, Is_features):
            # fc_label = cluster_features(fc)
            # weights = torch.ones_like(fc_label).float()
            # mean1, _= calc_mean_std(fc[0][fc_label==0].unsqueeze(0))
            # mean2, _= calc_mean_std(fc[0][fc_label==1].unsqueeze(0))
            # mean3, _= calc_mean_std(fc[0][fc_label==2].unsqueeze(0))
            # LOGGER.debug(mean1.mean())
            # LOGGER.debug(mean2.mean())
            # LOGGER.debug(mean3.mean())
            # _, ind = torch.sort(torch.tensor([mean1.mean(), mean2.mean(), mean3.mean()]))
            # vals = torch.gather(torch.tensor([0.3, 0.6, 1.0]), -1, ind)
            # LOGGER.debug(vals)
            # weights[fc_label==0] = vals[0]
            # weights[fc_label==1] = vals[1]
            # weights[fc_label==2] = vals[2]
            # LOGGER.debug(weights)
            # all_layer_weights.append(weights.to(self.device))
            # continue

            c_mean, c_std = calc_mean_std(fc)
            s_mean, s_std = calc_mean_std(fs)
            mean_diff_cs = abs(c_mean - s_mean).squeeze()
            _, indices = torch.sort(s_mean.squeeze())
            weights = torch.ones_like(indices).float()
            quater = len(indices) // 4
            # 1.0 0.8 0.2 0.1
            weights[indices[:quater]] = 0.5
            weights[indices[quater:quater*2]] = 0.5
            weights[indices[quater*2:quater*3]] = 1
            weights[indices[quater*3:]] = 1
            # weights[indices[:quater*4]] = 0
            # weights[indices[quater*4-4:]] = 1
            # weights[: quater*2] = 0
            # weights = 1 - torch.linspace(0.3, 0.9, len(indices)) ** 2
            all_layer_weights.append(weights.to(self.device))

        # Fc_indices = []
        # for Fc in Ic_features:
        #     B, C, W, H = Fc.shape
        #     _, index_content = torch.sort(Fc.view(B, C, -1))
        #     inverse_index = index_content.argsort(-1)
        #     # value_style.gather(-1, inverse_index)
        #     Fc_indices.append(inverse_index)

        Ic_features_masks = []
        for Fc in Ic_features:
            Fmask = Fc.clone() 
            Fmask[Fmask != 0] = 1
            LOGGER.debug(Fmask.sum()/Fc.numel())
               
            Ic_features_masks.append(Fmask)

        # optimize
        opt_img.requires_grad = True
        optimizer = optim.LBFGS([opt_img])

        n_iter = [0]
        while n_iter[0] <= self.config.max_iter:
            def closure():
                loss = 0
                optimizer.zero_grad()
                nonlocal Is_features

                # 提取当前特征
                Io_41, Io_feats = self.model(opt_img, hypercolumn=False)
                # Io_41, Is_features = self.model(self.lerp(Is, opt_img, 0.2), hypercolumn=False)

                # 计算损失
                #! Gram loss
                if self.check_method('gatys'):
                    loss = gram_loss(Io_feats, Is_features, div=True)
                elif self.check_method('gatysdivc'):
                    loss = gram_loss(Io_feats, Is_features, div=False)
                    # content_loss = torch.mean((Io_41 - Ic_41) ** 2)
                    # total_variance_loss = tv_loss(opt_img)
                    # loss += style_loss #+ 1e0 * min_d# + content_loss + total_variance_loss

                #! align mean std cov, ss, efdm
                if 'gatys' not in self.config.method:
                    for Fo, Fs, Fc, mask, weights in zip(Io_feats, Is_features, Ic_features, Ic_features_masks, all_layer_weights):
                        c_mean, c_std = calc_mean_std(Fo)
                        s_mean, s_std = calc_mean_std(Fs)
                        if self.check_method('l2'):
                            loss += torch.mean((Fo - Fs)**2)
                        if self.check_method('mean'):
                            loss += torch.mean((c_mean - s_mean)**2)
                        if self.check_method('std'):
                            loss += torch.mean((c_std - s_std)**2)
                        if self.check_method('adain'):
                            loss += torch.mean((c_mean - s_mean)**2 + (c_std - s_std)**2) 
                        if self.check_method('cov'):
                            if self.config.add_weights:
                                loss += covariance_loss(Fo * weights[:, None, None], Fs * weights[:, None, None])
                            else:
                                loss += covariance_loss(Fo, Fs)
                                # loss += covariance_loss(Fo * mask, Fs * mask)
                        if self.check_method('efdm'):
                            #! 优化时不需要先得到目标特征
                            #! EFDM算法就是用内容图的序，排列风格图的特征。
                            B, C, W, H = Fo.shape
                            _, index_content = torch.sort(Fo.view(B, C, -1))
                            value_style, _ = torch.sort(Fs.view(B, C, -1))
                            inverse_index = index_content.argsort(-1)
                            loss += torch.mean((Fo.view(B,C,-1)-value_style.gather(-1, inverse_index))**2)
                        if self.check_method('ss'):
                            Fo = self.downsample_features_if_needed(Fo)
                            Fs = self.downsample_features_if_needed(Fs)
                            Fc = self.downsample_features_if_needed(Fc)
                            loss += calc_ss_loss(Fo, Fc)

                        if self.config.visualize_grad:
                            Fo.retain_grad()

                loss.backward()
                
                # 日志
                n_iter[0] += 1
                if n_iter[0] < 50:
                    self.save_image(opt_img, f'result_{n_iter[0]}', verbose=True)

                if n_iter[0] == 1 or n_iter[0] % self.config.show_iter == 0:
                    if n_iter[0] == 1 and self.config.visualize_features:
                        self.visualize_features(Io_feats, 'Content')
                        self.visualize_features(Is_features, 'Style')
                    if n_iter[0] == self.config.max_iter and self.config.visualize_features:
                        LOGGER.info('Vis Ics')
                        self.visualize_features(Io_feats, 'Opt Ics')
                    
                    if self.config.visualize_grad:
                        self.visualize_features([Fo.grad], f'{self.config.layers}_Grad_{n_iter[0]}')
                        self.visualize_features([opt_img.grad], f'{self.config.layers}_Image_Grad_{n_iter[0]}')

                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')

                    self.save_image(opt_img, f'result_{n_iter[0]}', verbose=True)
                return loss

            # opt_img_old = opt_img.detach().clone()
            optimizer.step(closure)
            # self.save_image(opt_img.detach() - opt_img_old, f'update{n_iter[0]}', verbose=True)
            
        # save final image
        self.save_image(opt_img, 'result')
    
    def pyramid_optimize_process(self, Ic, Is):
        # 构建图像金字塔
        # Ic = (torch.rand_like(Ic) + torch.ones_like(Ic))/2
        Ic_pyramid = dec_lap_pyr(Ic, self.config.pyramid_levels)
        Is_pyramid = dec_lap_pyr(Is, self.config.pyramid_levels)
        Io_pyramid = dec_lap_pyr(Ic, self.config.pyramid_levels)

        LOGGER.info(len((Io_pyramid)))

        # 低分辨率时直接用内容初始化，高分辨率细节从0开始学习。
        # for i in range(1):
        #     Io_pyramid[i] *= 0

        # Stylize from coarse to fine
        for scale in range(self.config.max_scales)[::-1]:
            # 合成当前分辨率的内容图，风格图，输出图
            Is_scale = syn_lap_pyr(Is_pyramid[scale:])
            # Ic_scale = syn_lap_pyr(Ic_pyramid[scale:])
            # Io_scale = syn_lap_pyr(Io_pyramid[scale:])
            LOGGER.info(f'Stage {self.config.max_scales - scale}: {Is_scale.shape}')

            opt_vars = [torch.nn.Parameter(level) for level in Io_pyramid[scale:]]
            optimizer = torch.optim.Adam(opt_vars, lr=self.config.lr)

            _, Is_scale_features = self.model(Is_scale, hypercolumn=False, normalize=False)

            #! 优化过程
            n_iter = [0]
            output_image = None
            while n_iter[0] <= self.config.max_iter:
                def closure():
                    loss = 0
                    optimizer.zero_grad()
                    nonlocal output_image

                    # 合成当前尺度的图像
                    output_image = syn_lap_pyr(opt_vars)
                    # 提取当前输出特征
                    _, Io_scale_features = self.model(output_image, hypercolumn=False, normalize=False)
                    # 计算损失
                    for Fo, Fs in zip(Io_scale_features, Is_scale_features):
                        c_mean, c_std = calc_mean_std(Fo)
                        s_mean, s_std = calc_mean_std(Fs)
                        if scale == 0 and self.check_method('ours'):
                            # 如果特征图宽高太大，则进行稀疏采样
                            Fs = self.downsample_features_if_needed(Fs)
                            Fo = self.downsample_features_if_needed(Fo)
                            
                            if n_iter[0] == 0:
                                LOGGER.debug(f'Shape of features: {Fs.shape} {Fo.shape}')
                            
                            # 计算损失
                            if True:
                                loss += calc_remd_loss(Fs, Fo)
                            else:
                                loss += nnst_fs_loss(Fs, Fo)

                        if scale != 0 and self.check_method('ours'):
                            loss += covariance_loss(Fo, Fs)
                    
                    loss.backward()

                    # 日志
                    n_iter[0] += 1
                    if n_iter[0] % self.config.show_iter == 0:
                        LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')

                    return loss

                optimizer.step(closure)

            # 用优化好的图像重新构建图像金字塔，替换原输出图像金字塔的对应部分
            Io_pyramid[scale:] = dec_lap_pyr(output_image, self.config.pyramid_levels - scale)

            with torch.no_grad(): # 保存中间尺度优化结果
                output_image_tensor = syn_lap_pyr(Io_pyramid)
                self.save_image(output_image_tensor, f'pyr_{scale}', verbose=True)
                self.save_image(F.interpolate(syn_lap_pyr(opt_vars), (512, 512)), f'pyr_inter_{scale}', verbose=True)
                # self.writer.log_image(f'pyr_{scale}', self.tensor2pil(output_image_tensor))
        
        with torch.no_grad(): # 保存最终输出
            output_image_tensor = syn_lap_pyr(Io_pyramid)
            self.save_image(output_image_tensor, 'result')
            # self.writer.log_image('result', self.tensor2pil(output_image_tensor))


if __name__ == '__main__':
    import os
    args = tyro.cli(GatysConfig)
    pipeline = GatysPipeline(args)

    # content_dir='data/contents/'
    # style_dir='data/styles/'
    # for c in os.listdir(content_dir):
    #     for s in os.listdir(style_dir):
    #         pipeline(
    #             content_dir + c,
    #             style_dir + s
    #         )
    # exit()

    # content_dir='data/content/'
    # style_dir='data/nnst_style/'
    # for s in os.listdir(style_dir):
    #     pipeline(
    #         'data/content/sailboat.jpg', 
    #         style_dir + s
    #     )
    #     # break
    # exit()

    pipeline(
        args.content_path,
        args.style_path
    )
    