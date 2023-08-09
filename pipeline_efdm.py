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
    # layers: str = "0,5,10,17,24"
    layers: str = "1,6,11,18,25"
    """layer indices of style features which should seperate by ','"""
    optimizer: str = 'lbfgs'
    """optimizer type: adam | lbfgs"""
    max_iter: int = 500
    exact_resize: bool = True


class EFDMPipeline(OptimzeBasePipeline):
    def __init__(self, config: EFDMConfig) -> None:
        super().__init__(config)
    
    def add_extra_infos(self):
        infos =[]
        return infos
    
    def common_optimize_process(self, Ic, Is):
        # opt_img = torch.rand_like(content_image)
        opt_img = Ic.data.clone()
        opt_img = opt_img.to(self.device)

        # save init image
        self.save_image(opt_img, 'init')

        # get target features
        _, Is_features = self.model(Is)
        Ic_41, Ic_features = self.model(Ic)

        #! 直接用内容图的风格图的特征进行EFDM匹配得到的特征作为目标来优化
        #! 结果有很多噪点，内容结构保存较好，颜色更偏风格图。
        target_features = [exact_feature_distribution_matching(Ic_features[i], Is_features[i]) for i in range(len(Is_features))]

        # optimize
        opt_img.requires_grad = True
        optimizer = optim.LBFGS([opt_img], lr=1)
        n_iter = [0]

        while n_iter[0] <= self.config.max_iter:
            def closure():
                loss = 0
                optimizer.zero_grad()

                _, Io_feats = self.model(opt_img)

                style_loss = 0
                for Fo, Fs in zip(Io_feats, Is_features):
                    #! 优化时不需要先得到目标特征
                    #! EFDM算法就是用内容图的序，排列风格图的特征。
                    B, C, W, H = Fo.shape
                    _, index_content = torch.sort(Fo.view(B, C, -1))
                    value_style, _ = torch.sort(Fs.view(B, C, -1))
                    inverse_index = index_content.argsort(-1)
                    style_loss += torch.mean((Fo.view(B,C,-1)-value_style.gather(-1, inverse_index))**2)

                # total_variance_loss = tv_loss(opt_img)
                loss += self.config.lambda_s * style_loss# + cov_loss# + content_loss #+ 1 * total_variance_loss
 
                loss.backward()

                n_iter[0] += 1
                # print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')
                    # LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                    self.save_image(opt_img, str(n_iter[0]//self.config.show_iter), verbose=True)
                    
                return loss
            
            optimizer.step(closure)
        
        # save final image
        self.save_image(opt_img, 'result')


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
        'data/style/17.jpg',
        # 'data/nnst_style/shape.png',
    )
