from dataclasses import dataclass

import torch
import tyro
from einops import rearrange, repeat
from PIL import Image
from torch import optim

from losses import gram_loss, tv_loss
from utils import CONSOLE, AorB, check_folder
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline



@dataclass
class GatysConfig(OptimzeBaseConfig):
    """Gatys NST Arguments  
    IN 改变了特征响应的均值和方差  
    input_range 也改变了响应的均值和方差  
    standrardize 只改变了均值
    """

    output_dir: str = "./results/gatys"
    """output dir"""
    lambda_s: float = 1000
    """style weight"""
    use_tv_loss: bool = False
    """use total variance loss"""


class GatysPipeline(OptimzeBasePipeline):
    def __init__(self, config: GatysConfig) -> None:
        super().__init__(config)
        CONSOLE.print(config)
        self.config = config
        check_folder(self.config.output_dir)
    
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
            erase_mask = torch.zeros_like(opt_img[0, 0])
            erase_mask[0:300, 0:300] = 1
            erase_mask = (erase_mask == 1)
            erase_mask = erase_mask[None, None, ...]
            self.erase_mask = repeat(erase_mask, 'b c h w -> b (r c) h w', r=3)
            # self.erase_mask = torch.rand_like(opt_img) > 0.4

            opt_img[self.erase_mask] = content_image[self.erase_mask]

        # save init image
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimze_images.append(out_img)

        opt_img.requires_grad = True

        # get target features
        _, style_features = self.vgg.forward(style_image)
        content_features, fs = self.vgg.forward(content_image)
        if self.config.verbose: # 查看各层特征图的实际内容
            self.vis_feature_activations()

        # optimize
        optimizer = optim.LBFGS([opt_img])
        # style_weights = [n/256 for n in [64,128,256,512,512]]
        style_weights = [128/n for n in [64,128,256,512,512]]
        n_iter = [0]

        while n_iter[0] <= self.config.max_iter:
            def closure():
                optimizer.zero_grad()
                c_feats, s_feats = self.vgg(opt_img)
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
                    CONSOLE.print(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                    out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                    optimze_images.append(out_img)
                return loss
            
            optimizer.step(closure)
        
        # save final image
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimze_images.append(out_img)
        
        return optimze_images

    
if __name__ == '__main__':
    args = tyro.cli(GatysConfig)
    pipeline = GatysPipeline(args)
    pipeline(
        'data/content/sailboat.jpg', 
        'data/style/19.jpg',
        mask=False
    )
