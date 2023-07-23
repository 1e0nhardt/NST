from dataclasses import dataclass

import torch
import tyro
from einops import rearrange, repeat
from PIL import Image
from torch import optim

from utils import LOGGER, check_folder, AorB
from utils_st import exact_feature_distribution_matching
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline


@dataclass
class EFDMConfig(OptimzeBaseConfig):
    """EFDM NST Arguments  
    IN 改变了特征响应的均值和方差  
    input_range 也改变了响应的均值和方差  
    standrardize 只改变了均值
    """

    output_dir: str = "./results/efdm"
    """output dir"""
    lambda_s: float = 100
    """style weight"""
    use_tv_loss: bool = False
    """use total variance loss"""


class EFDMPipeline(OptimzeBasePipeline):
    def __init__(self, config: EFDMConfig) -> None:
        super().__init__(config)
        LOGGER.info(config)
        self.config = config
        check_folder(self.config.output_dir)
    
    def add_extra_infos(self):
        infos = []
        infos += [str(self.config.lambda_s)]
        return infos
    
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
        target_features = [exact_feature_distribution_matching(fs[i], style_features[i]) for i in range(len(style_features)-1)] # 4-1

        # optimize
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]

        while n_iter[0] <= self.config.max_iter:
            def closure():
                optimizer.zero_grad()

                _, s_feats = self.model(opt_img)

                style_loss = 0
                for i in range(len(s_feats) - 1):
                    input = s_feats[i]
                    target = target_features[i]
                    B, C, W, H = input.size(0), input.size(1), input.size(2), input.size(3)
                    _, index_content = torch.sort(input.view(B, C, -1))
                    value_style, _ = torch.sort(target.view(B, C, -1))
                    inverse_index = index_content.argsort(-1)
                    style_loss += torch.mean((input.view(B,C,-1)-value_style.gather(-1, inverse_index))**2)

                content_loss = torch.mean((s_feats[-2] - target_features[-1]) ** 2)

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
                    out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                    optimize_images.append(out_img)
                return loss
            
            optimizer.step(closure)
        
        # save final image
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images.append(out_img)
        
        return optimize_images


if __name__ == '__main__':
    args = tyro.cli(EFDMConfig)
    pipeline = EFDMPipeline(args)
    import os
    for c in os.listdir('data/content'):
            for s in [0, 6, 8, 122, 130, 143]:
                pipeline(
                    f'data/content/{c}',
                    f'data/style/{s}.jpg',
                    mask=False
                )
    # pipeline(
    #     'data/content/sailboat.jpg', 
    #     'data/style/6.jpg',
    #     mask=False
    # )
