from dataclasses import dataclass

import torch
import tyro
from einops import rearrange, repeat
from PIL import Image
from torch import optim

from losses import gram_loss, tv_loss
from nnst.image_pyramid import dec_lap_pyr, syn_lap_pyr
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from utils import CONSOLE, LOGGER, AorB, check_folder, generate_perlin_noise_2d
from utils_st import calc_mean_std


@dataclass
class GatysConfig(OptimzeBaseConfig):
    """Gatys NST Arguments  
    IN 改变了特征响应的均值和方差  
    input_range 也改变了响应的均值和方差  
    standrardize 只改变了均值
    """

    name: str = 'gatys'
    lambda_s: float = 1
    """style weight"""
    optimizer: str = 'lbfgs'
    """optimizer type: adam | lbfgs"""
    use_lap_pyr: bool = False


class GatysPipeline(OptimzeBasePipeline):
    def __init__(self, config: GatysConfig) -> None:
        super().__init__(config)
    
    def add_extra_infos(self):
        return AorB(self.config.use_lap_pyr, 'pyr', 'rgb') + ['cinit_withoutcontentloss_adain']
    
    def common_optimize_process(self, Ic, Is):
        opt_img = Ic.data.clone()
        # opt_img = torch.rand_like(Ic)
        # opt_img = torch.from_numpy(generate_perlin_noise_2d(Ic.shape[2:], (4, 4))).unsqueeze(0).unsqueeze(0).float()
        # opt_img = repeat(opt_img, 'n c h w -> n (a c) h w', a=3)
        opt_img = opt_img.to(self.device)

        # save init image
        self.save_image(opt_img, 'init')

        # get target features
        _, Is_features = self.model.forward(Is)
        Ic_41, Ic_features = self.model.forward(Ic)

        # optimize
        opt_img.requires_grad = True
        optimizer = optim.LBFGS([opt_img])
            
        n_iter = [0]
        while n_iter[0] <= self.config.max_iter:
            def closure():
                loss = 0
                optimizer.zero_grad()

                Io_41, Io_feats = self.model(opt_img)

                #! Gram loss
                style_loss = gram_loss(Io_feats, Is_features)
                # content_loss = torch.mean((Io_41 - Ic_41) ** 2)
                # total_variance_loss = tv_loss(opt_img)
                loss += style_loss# + content_loss #+ 1 * total_variance_loss

                #! align mean and std
                # for cur, style in zip(Io_feats, Is_features):
                #     c_mean, c_std = calc_mean_std(cur)
                #     s_mean, s_std = calc_mean_std(style)
                #     # loss += torch.mean((c_mean - s_mean)**2) 
                #     # loss += torch.mean((c_std - s_std)**2) 
                #     loss += torch.mean((c_mean - s_mean)**2 + (c_std - s_std)**2) 
                
                #! kl loss | cross entropy loss
                # for cur, style in zip(s_feats, style_features):
                #     style = softmax_smoothing_2d(torch.sigmoid(style))
                #     cur = softmax_smoothing_2d(torch.sigmoid(cur))
                #     entropy_reg_loss = 1e10 * (style * (torch.log(style) - torch.log(cur))).mean()
                #     loss += entropy_reg_loss
                # loss = style_loss# + content_loss

                loss.backward()

                #* 将不需要优化的区域的梯度置为零
                # if mask:
                #     opt_img.grad[~self.erase_mask] = 0

                n_iter[0] += 1
                #print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')
                    # LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                return loss
            
            optimizer.step(closure)
        
        # save final image
        self.save_image(opt_img, 'result')


if __name__ == '__main__':
    args = tyro.cli(GatysConfig)
    pipeline = GatysPipeline(args)
    pipeline(
        'data/content/sailboat.jpg', 
        # 'results/gatys/vgg19_lbfgs_std_R1_1_rgb__withcontentloss/sailboat_S4_result_S4_result.png', 
        # 'results/gatys/vgg19_lbfgs_std_R1_1_rgb_alignmeanstd_withcontentloss/sailboat_S4_result_S4_result_S4_result_S4_result.png', 
        # 'data/style/130.jpg',
        'data/nnst_style/S1.jpg',
        # 'data/nnst_style/circles.png',
    )
    