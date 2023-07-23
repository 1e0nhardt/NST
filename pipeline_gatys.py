from dataclasses import dataclass

import torch
import tyro
from einops import rearrange, repeat
from PIL import Image
from torch import optim

from losses import gram_loss, tv_loss
from utils import LOGGER, AorB, check_folder
from utils_st import softmax_smoothing, get_gaussian_conv
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
import math

@dataclass
class GatysConfig(OptimzeBaseConfig):
    """Gatys NST Arguments  
    IN 改变了特征响应的均值和方差  
    input_range 也改变了响应的均值和方差  
    standrardize 只改变了均值
    """

    output_dir: str = "./results/gatys"
    """output dir"""
    lambda_s: float = 1e2
    """style weight"""
    use_tv_loss: bool = False
    """use total variance loss"""
    use_softmax_smoothing: bool = False
    use_conv_smoothing: bool = False


class GatysPipeline(OptimzeBasePipeline):
    def __init__(self, config: GatysConfig) -> None:
        super().__init__(config)
        LOGGER.info(config)
        self.config = config
        check_folder(self.config.output_dir)
    
    def add_extra_infos(self):
        return [str(self.config.lambda_s)] + AorB(self.config.use_conv_smoothing, 'ConvSmooth')
    
    def optimize_process(self, content_path, style_path, mask=False):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        # convert("RGB")会让灰度图像也变为3通道
        style_image = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)

        # optimize target
        optimize_images = []
        opt_img = content_image.data.clone()
        opt_img = opt_img.to(self.device)

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
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images.append(out_img)

        opt_img.requires_grad = True

        # get target features
        _, style_features = self.model.forward(style_image)
        content_features, fs = self.model.forward(content_image)

        g_convs = [get_gaussian_conv(n, 3, 5).to(self.device) for n in [64,128,256,512,512]]
        if self.config.use_conv_smoothing:
            for i in range(len(style_features)):
                style_features[i] = g_convs[i](style_features[i])

        if self.config.use_softmax_smoothing:
            # scale = []
            # for m in style_features:
            #     m = m.detach()
            #     logger.info('Feature Before Smoothing: mean=%.3e, var=%.3e, max=%.3e, min=%.3e, >mean_percent=%.4f', 
            #     m.mean().item(), m.var().item(), m.max().item(), m.min().item(),
            #     (m > m.mean()).float().sum() / math.prod(opt_img.shape))
            #     scale.append(m[m > 0.1].mean())
            # torch.save(style_features, 'SFs.pt')
            style_features = softmax_smoothing(style_features)
            # torch.save(style_features, 'SFSs.pt')
            # for i in range(len(style_features)):
            #     m = style_features[i]
            #     style_features[i] = style_features[i] * (scale[i] / m.detach().mean())
            #     logger.info(scale[i] / m.detach().mean())
            #     logger.info('Feature After Smoothing: mean=%.3e, var=%.3e, max=%.3e, min=%.3e, >mean_percent=%.4f', 
            #     m.detach().mean().item(), m.detach().var().item(), m.detach().max().item(), m.detach().min().item(),
            #     (m > m.detach().mean()).float().sum() / math.prod(opt_img.shape))

        # optimize
        if self.config.optimizer == 'lbfgs':
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
                c_feats, s_feats = self.model(opt_img)

                if self.config.use_conv_smoothing:
                    for i in range(len(s_feats)):
                        s_feats[i] = g_convs[i](s_feats[i])

                # logger.info(s_feats[2].sum((2, 3)), style='green')
                if self.config.use_softmax_smoothing:
                    # scale = []
                    # for m in s_feats:
                        # logger.info('Feature Before Smoothing: %.3e, %.3e, %.3e, %.3e', m.detach().mean().item(), m.detach().var().item(), m.detach().max().item(), m.detach().min().item())
                        # scale.append(m.detach().mean())
                    s_feats = softmax_smoothing(s_feats)
                    # for i in range(len(s_feats)):
                    #     m = s_feats[i]
                        # logger.info('Feature Afetr Smoothing: %.3e, %.3e, %.3e, %.3e', m.detach().mean().item(), m.detach().var().item(), m.detach().max().item(), m.detach().min().item())
                        # s_feats[i] = s_feats[i] * (scale[i] / m.detach().mean())
                    # logger.info(s_feats[2].sum((2, 3)), style='yellow')
                style_loss = gram_loss(s_feats, style_features, style_weights)
                content_loss = torch.mean((c_feats - content_features) ** 2)
                # total_variance_loss = tv_loss(opt_img)
                loss = self.config.lambda_s * style_loss + content_loss #+ 1 * total_variance_loss
                loss.backward()
                # 将不需要优化的区域的梯度置为零
                if mask:
                    opt_img.grad[~self.erase_mask] = 0
                n_iter[0] += 1

                #print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                    LOGGER.info(
                        'Grad of x: mean=%.3e, var=%.3e, max=%.3e, min=%.3e, large_percent=%.4f', 
                        opt_img.grad.mean().item(), opt_img.grad.var().item(), 
                        opt_img.grad.max().item(), opt_img.grad.min().item(),
                        (opt_img.grad > opt_img.detach().mean()).float().sum() / math.prod(opt_img.shape))
                    out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                    optimize_images.append(out_img)
                return loss
            
            # optimizer.step(closure)
            optimizer.zero_grad()
            c_feats, s_feats = self.model(opt_img)
            if self.config.use_conv_smoothing:
                for i in range(len(s_feats)):
                    s_feats[i] = g_convs[i](s_feats[i])

            # logger.info(s_feats[2].sum((2, 3)), style='green')
            if self.config.use_softmax_smoothing:
                # scale = []
                # for m in s_feats:
                    # logger.info('Feature Before Smoothing: %.3e, %.3e, %.3e, %.3e', m.detach().mean().item(), m.detach().var().item(), m.detach().max().item(), m.detach().min().item())
                    # scale.append(m.detach().mean())
                s_feats = softmax_smoothing(s_feats)
                # for i in range(len(s_feats)):
                #     m = s_feats[i]
                    # logger.info('Feature Afetr Smoothing: %.3e, %.3e, %.3e, %.3e', m.detach().mean().item(), m.detach().var().item(), m.detach().max().item(), m.detach().min().item())
                    # s_feats[i] = s_feats[i] * (scale[i] / m.detach().mean())
                # logger.info(s_feats[2].sum((2, 3)), style='yellow')

            style_loss = gram_loss(s_feats, style_features, style_weights)
            content_loss = torch.mean((c_feats - content_features) ** 2)
            # total_variance_loss = tv_loss(opt_img)
            loss = self.config.lambda_s * style_loss + content_loss #+ 1 * total_variance_loss
            loss.backward()

            # 将不需要优化的区域的梯度置为零
            if mask:
                opt_img.grad[~self.erase_mask] = 0
            n_iter[0] += 1

            #print loss
            if n_iter[0] % self.config.show_iter == 0:
                LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                LOGGER.info(
                    'Grad of x: mean=%.3e, var=%.3e, max=%.3e, min=%.3e, large_percent=%.4f', 
                    opt_img.grad.mean().item(), opt_img.grad.var().item(), 
                    opt_img.grad.max().item(), opt_img.grad.min().item(),
                    (opt_img.grad > opt_img.grad.mean()).float().sum() / math.prod(opt_img.shape))
                out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                optimize_images.append(out_img)
            
            optimizer.step()
        
        # save final image
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images.append(out_img)
        
        return optimize_images

    
if __name__ == '__main__':
    args = tyro.cli(GatysConfig)
    pipeline = GatysPipeline(args)
    pipeline(
        'data/content/sailboat.jpg', 
        'data/style/17.jpg',
        mask=False
    )
    