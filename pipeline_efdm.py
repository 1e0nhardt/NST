from dataclasses import dataclass

import torch
import tyro
from PIL import Image
from torch import optim

from losses import gram_loss
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from utils import LOGGER, AorB, check_folder
from utils_st import exact_feature_distribution_matching


@dataclass
class EFDMConfig(OptimzeBaseConfig):
    """EFDM NST Arguments  
    IN 改变了特征响应的均值和方差  
    input_range 也改变了响应的均值和方差  
    standrardize 只改变了均值
    """

    name: str = 'efdm'
    """pipeline name."""
    lambda_s: float = 1
    """style weight"""
    model_type: str = 'vgg16'
    """feature extractor model type vgg16 | vgg19 | inception"""
    # layers: str = "1,6,11,20,29"
    # layers: str = "1"
    layers: str = "1,6,11,18,25"
    """layer indices of style features which should seperate by ','"""
    optimizer: str = 'lbfgs'
    """optimizer type: adam | lbfgs"""
    max_iter: int = 500


class EFDMPipeline(OptimzeBasePipeline):
    def __init__(self, config: EFDMConfig) -> None:
        super().__init__(config)
        self.config = config
        check_folder(self.config.output_dir)
    
    def add_extra_infos(self):
        infos += ['gramreg']
        return infos
    
    def optimize_process(self, content_path, style_path):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        # convert("RGB")会让灰度图像也变为3通道
        style_image = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)
        # img = content_image
        # content_image = style_image
        # style_image = img

        # opt_img = torch.rand_like(content_image)
        opt_img = content_image.data.clone()
        opt_img = opt_img.to(self.device)

        # save init image
        self.optimize_images['init'] = self.tensor2pil(opt_img)

        opt_img.requires_grad = True

        # get target features
        _, style_features = self.model.forward(style_image)
        content_features, fs = self.model.forward(content_image)

        #! 直接用内容图的风格图的特征进行EFDM匹配得到的特征作为目标来优化
        #! 结果有很多噪点，内容结构保存较好，颜色更偏风格图。
        target_features = [exact_feature_distribution_matching(fs[i], style_features[i]) for i in range(len(style_features))]

        # optimize
        # optimizer = optim.Adam([opt_img], lr=1e-2)
        optimizer = optim.LBFGS([opt_img], lr=1)
        n_iter = [0]

        while n_iter[0] <= self.config.max_iter:
            def closure():
                optimizer.zero_grad()

                _, o_feats = self.model(opt_img)

                style_loss = 0
                for i in range(len(o_feats)):
                    #! 优化时不需要先得到目标特征
                    #! EFDM算法就是用内容图的序，排列风格图的特征。
                    input = o_feats[i]
                    target = style_features[i]
                    B, C, W, H = input.size(0), input.size(1), input.size(2), input.size(3)
                    _, index_content = torch.sort(input.view(B, C, -1))
                    value_style, _ = torch.sort(target.view(B, C, -1))
                    inverse_index = index_content.argsort(-1)
                    style_loss += torch.mean((input.view(B,C,-1)-value_style.gather(-1, inverse_index))**2)

                # total_variance_loss = tv_loss(opt_img)
                cov_loss = gram_loss(o_feats, style_features)

                loss = self.config.lambda_s * style_loss + cov_loss# + content_loss #+ 1 * total_variance_loss
 
                loss.backward()

                n_iter[0] += 1

                # print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')
                    # LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                    self.optimize_images[str(n_iter[0]//self.config.show_iter)] = self.tensor2pil(opt_img)
                return loss
            
            optimizer.step(closure)
        
        # save final image
        self.optimize_images['result'] = self.tensor2pil(opt_img)


if __name__ == '__main__':
    args = tyro.cli(EFDMConfig)
    pipeline = EFDMPipeline(args)
    import os

    # content_dir='data/content/'
    style_dir='data/nnst_style/'
    # for c in os.listdir(content_dir):
    #     for s in os.listdir(style_dir):
    #         pipeline(
    #             content_dir + c,
    #             style_dir + s
    #         )
    # exit()

    # for s in os.listdir(style_dir):
    #     pipeline(
    #         'data/content/sailboat.jpg',
    #         style_dir + s
    #     )

    pipeline(
        'data/content/sailboat.jpg', 
        'data/style/130.jpg',
        # 'data/nnst_style/shape.png',
    )
