from dataclasses import dataclass
import os

import torch
import tyro
from PIL import Image
from torch import optim

from losses import gram_loss, tv_loss, gram_matrix
from nnst.image_pyramid import dec_lap_pyr, syn_lap_pyr
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
from utils import LOGGER, AorB, check_folder
from utils_st import (calc_mean_std, exact_feature_distribution_matching, get_gaussian_conv,
                      softmax_smoothing, softmax_smoothing_2d)


@dataclass
class ExprConfig(OptimzeBaseConfig):

    name: str = 'explore'
    model_type: str = 'vgg16'
    """feature extractor model type vgg16 | vgg19 | inception"""
    # layers: str = "25"
    layers: str = "1,6,11,18,25"
    # layers: str = "25, 27, 29, 22, 20, 18, 15, 13, 11, 8, 6, 3, 1"
    """layer indices of style features which should seperate by ','"""
    optimizer: str = 'lbfgs'
    """optimizer type: adam | lbfgs"""
    max_iter: int = 500
    save_process: bool = False
    use_tv_reg: bool = False
    save_init: bool = True
    exact_resize: bool = True


class ExprPipeline(OptimzeBasePipeline):
    def __init__(self, config: ExprConfig) -> None:
        super().__init__(config)
    
    def add_extra_infos(self):
        return [f'reverse_content'] + AorB(self.config.use_tv_reg, '10tvloss') + ['adaintarget']
    
    def add_extra_file_infos(self):
        return ['cont-lap']
    
    def common_optimize_process(self, Ic, Is):
        opt_img = Ic.data.clone()
        # opt_img = Is.data.clone()
        # opt_img = torch.rand_like(Ic)
        opt_img = opt_img.to(self.device)

        # output_pyramid = dec_lap_pyr(Ic.clone(), 8)

        # save init image
        self.save_image(opt_img, 'init')

        opt_img.requires_grad = True

        # get target features
        _, Is_features = self.model(Is)
        Ic_41, Ic_features = self.model(Ic)

        _, laplacian_features = self.model(self.get_lap_filtered_image(self.content_path))

        # Is_features = Ic_features

        # get target features
        # target_features = [exact_feature_distribution_matching(Ic_features[i], Is_features[i]) for i in range(len(Is_features)-1)] # efdm
        target_features = []
        for cur, lap in zip(Ic_features, laplacian_features):
            target_features.append(cur - lap)

        # for cur, style in zip(Ic_features, Is_features):
        #     c_mean, c_std = calc_mean_std(cur)
        #     s_mean, s_std = calc_mean_std(style)
        #     LOGGER.debug(f'c_mean={c_mean}, c_std={c_std}, s_mean={s_mean}, s_std={s_std}')
        #     target_features.append(((cur - c_mean)/c_std) * s_std + s_mean)
                    
        # target_features = [f + torch.ones_like(f) for f in Ic_features]
        # target_features = [f + f/2 * torch.rand_like(f) for f in Ic_features]


        #! 特征插值
        # for i in range(len(Is_features)):
        #     Is_features[i] = (Is_features[i] + Ic_features[i])/2

        # optimize
        # opt_vars = [torch.nn.Parameter(level_image) for level_image in output_pyramid]

        # optimizer = optim.LBFGS(opt_vars, lr=0.1)
        optimizer = optim.LBFGS([opt_img], lr=1)

        n_iter = [0]
        while n_iter[0] <= self.config.max_iter:
            def closure():
                loss = 0
                optimizer.zero_grad()

                # opt_img = syn_lap_pyr(opt_vars)
                Io_41, Io_feats = self.model(opt_img)

                # content_loss = torch.mean((Io_41 - content_features) ** 2)
                
                total_variance_loss = tv_loss(opt_img)

                # for cur, style in zip(s_feats, Is_features):
                for cur, style in zip(Io_feats, target_features):
                    loss += torch.mean((cur - style) ** 2)
                    # 交叉熵
                    # style = softmax_smoothing_2d(torch.sigmoid(style))
                    # cur = softmax_smoothing_2d(torch.sigmoid(cur))
                    # entropy_reg_loss = 1e10 * (style * (torch.log(style) - torch.log(cur))).mean()
                    # loss += entropy_reg_loss
                
                if self.config.use_tv_reg:
                    loss += 10 * total_variance_loss

                loss.backward()

                n_iter[0] += 1

                #print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')
                    self.save_image(opt_img, str(n_iter[0]//self.config.show_iter), verbose=True)
                return loss
            
            optimizer.step(closure)
        
        # save final image
        # opt_img = syn_lap_pyr(opt_vars)
        self.save_image(opt_img, 'result')
        
    
if __name__ == '__main__':
    args = tyro.cli(ExprConfig)
    pipeline = ExprPipeline(args)
    content_dir='data/content/'
    # style_dir='data/nnst_style/'
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
    
    for c in os.listdir(content_dir)[:5]:
        pipeline(
            content_dir + c,
            'data/nnst_style/S4.jpg'
        )

    # pipeline(
    #     'data/content/C1.png', 
    #     # 'data/content/sailboat.jpg', 
    #     'data/nnst_style/S4.jpg'
    # )
    