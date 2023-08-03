from collections import OrderedDict
from dataclasses import dataclass

import torch
import tyro
from einops import rearrange, repeat
from PIL import Image
from torch import optim

from losses import gram_loss, tv_loss
from nnst.image_pyramid import dec_lap_pyr, syn_lap_pyr
from utils import LOGGER, AorB, check_folder, generate_perlin_noise_2d
from utils_st import softmax_smoothing, get_gaussian_conv, softmax_smoothing_2d, calc_mean_std
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
import math

@dataclass
class GatysConfig(OptimzeBaseConfig):
    """Gatys NST Arguments  
    IN 改变了特征响应的均值和方差  
    input_range 也改变了响应的均值和方差  
    standrardize 只改变了均值
    """

    name: str = 'gatys'
    output_dir: str = "./results/gatys"
    """output dir"""
    lambda_s: float = 1
    """style weight"""
    use_tv_loss: bool = False
    """use total variance loss"""
    use_softmax_smoothing: bool = False
    use_conv_smoothing: bool = False
    use_wandb: bool = False
    optimizer: str = 'lbfgs'
    """optimizer type: adam | lbfgs"""
    use_lap_pyr: bool = False


class GatysPipeline(OptimzeBasePipeline):
    def __init__(self, config: GatysConfig) -> None:
        super().__init__(config)
        self.config = config
        check_folder(self.config.output_dir)
    
    def add_extra_infos(self):
        return [str(self.config.lambda_s)] + AorB(self.config.use_conv_smoothing, 'ConvSmooth')+ AorB(self.config.use_lap_pyr, 'pyr', 'rgb') + ['rinit_withoutcontentloss_adain']
    
    def optimize_process(self, content_path, style_path, mask=False):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        # convert("RGB")会让灰度图像也变为3通道
        style_image = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)

        # optimize target
        optimize_images = OrderedDict()

        if self.config.use_lap_pyr:
            output_pyramid = dec_lap_pyr(content_image.clone(), 8)
        else:
            # opt_img = content_image.data.clone()
            opt_img = torch.rand_like(content_image)
            # opt_img = torch.from_numpy(generate_perlin_noise_2d(content_image.shape[2:], (4, 4))).unsqueeze(0).unsqueeze(0).float()
            # opt_img = repeat(opt_img, 'n c h w -> n (a c) h w', a=3)
            opt_img = opt_img.to(self.device)
            # out = (content_image.clone() + style_image.clone())/2

        if mask:
            opt_img = self.transform_pre(Image.open(
                self.generate_expr_name()+f"/{self.generate_filename(content_path, style_path)}_result.png"
            )).unsqueeze(0).to(self.device)

            # generate mask
            erase_mask = torch.zeros_like(opt_img[0, 0])
            erase_mask[0:300, 0:300] = 1
            erase_mask = (erase_mask == 1)
            erase_mask = erase_mask[None, None, ...]
            self.erase_mask = repeat(erase_mask, 'b c h w -> b (r c) h w', r=3)
            # self.erase_mask = torch.rand_like(opt_img) > 0.4

            opt_img[self.erase_mask] = content_image[self.erase_mask]

        # save init image
        # out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        # optimize_images['init'] = out_img

        # get target features
        _, style_features = self.model.forward(style_image)
        content_features, fs = self.model.forward(content_image)

        g_convs = [get_gaussian_conv(n, 3, 5).to(self.device) for n in [64,128,256,512,512]]
        if self.config.use_conv_smoothing:
            for i in range(len(style_features)):
                style_features[i] = g_convs[i](style_features[i])

        if self.config.use_softmax_smoothing:
            style_features = softmax_smoothing(style_features)

        # optimize
        if self.config.use_lap_pyr:
            opt_vars = [torch.nn.Parameter(level_image) for level_image in output_pyramid]
        else:
            opt_img.requires_grad = True

        if self.config.optimizer == 'lbfgs':
            if self.config.use_lap_pyr:
                optimizer = optim.LBFGS(opt_vars, lr=0.1)
            else:
                optimizer = optim.LBFGS([opt_img])
        elif self.config.optimizer == 'adam':
            optimizer = optim.Adam([opt_img], lr=self.config.lr)
        else: 
            raise RuntimeError(f'{self.config.optimizer} is not supported')
        # style_weights = [n/256 for n in [64,128,256,512,512]]
        style_weights = [1 for n in [64,128,256,512,512]]
        n_iter = [0]

        while n_iter[0] <= self.config.max_iter:
            def closure():
                optimizer.zero_grad()
                nonlocal opt_img
                if self.config.use_lap_pyr:
                    opt_img = syn_lap_pyr(opt_vars)

                c_feats, s_feats = self.model(opt_img)

                if self.config.use_conv_smoothing:
                    for i in range(len(s_feats)):
                        s_feats[i] = g_convs[i](s_feats[i])

                if self.config.use_softmax_smoothing:
                    s_feats = softmax_smoothing(s_feats)
                
                loss = 0
                #! Gram loss
                # style_loss = gram_loss(s_feats, style_features, style_weights)
                # content_loss = torch.mean((c_feats - content_features) ** 2)
                # total_variance_loss = tv_loss(opt_img)
                # loss = self.config.lambda_s * style_loss# + content_loss #+ 1 * total_variance_loss

                #! align mean and std
                for cur, style in zip(s_feats, style_features):
                    c_mean, c_std = calc_mean_std(cur)
                    s_mean, s_std = calc_mean_std(style)
                    loss += torch.mean((c_mean - s_mean)**2 + (c_std - s_std)**2) 
                
                #! kl loss | cross entropy loss
                # for cur, style in zip(s_feats, style_features):
                #     style = softmax_smoothing_2d(torch.sigmoid(style))
                #     cur = softmax_smoothing_2d(torch.sigmoid(cur))
                #     entropy_reg_loss = 1e10 * (style * (torch.log(style) - torch.log(cur))).mean()
                #     loss += entropy_reg_loss
                # loss = style_loss# + content_loss
                loss.backward()
                # 将不需要优化的区域的梯度置为零
                if mask:
                    opt_img.grad[~self.erase_mask] = 0
                n_iter[0] += 1

                #print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')
                    # LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                return loss
            
            optimizer.step(closure)
        
        # save final image
        if self.config.use_lap_pyr:
            opt_img = syn_lap_pyr(opt_vars)
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images['result'] = out_img
        
        return optimize_images

    
if __name__ == '__main__':
    args = tyro.cli(GatysConfig)
    pipeline = GatysPipeline(args)
    pipeline(
        'data/content/sailboat.jpg', 
        # 'results/gatys/vgg19_lbfgs_std_R1_1_rgb__withcontentloss/sailboat_S4_result_S4_result.png', 
        # 'results/gatys/vgg19_lbfgs_std_R1_1_rgb_alignmeanstd_withcontentloss/sailboat_S4_result_S4_result_S4_result_S4_result.png', 
        # 'data/style/130.jpg',
        'data/nnst_style/S4.jpg',
        # 'data/nnst_style/circles.png',
        mask=False
    )
    