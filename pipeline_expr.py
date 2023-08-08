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
        self.config = config
        check_folder(self.config.output_dir)
    
    def add_extra_infos(self):
        return [f'reverse_content'] + AorB(self.config.use_tv_reg, '10tvloss') + ['adaintarget']
    
    def add_extra_file_infos(self):
        return ['cont-lap']
    
    def optimize_process(self, content_path, style_path):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        # convert("RGB") makes grayscale image also has 3 channels.
        style_image = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)

        # style_image = (content_image.clone() + style_image.clone())/2
        opt_img = content_image.data.clone()
        # opt_img = style_image.data.clone()
        # opt_img = torch.rand_like(content_image)
        opt_img = opt_img.to(self.device)

        # output_pyramid = dec_lap_pyr(content_image.clone(), 8)

        # save init image
        self.optimize_images['init'] = self.tensor2pil(opt_img)

        opt_img.requires_grad = True

        # get target features
        _, style_features = self.model(style_image)
        content_features, fs = self.model(content_image)

        _, laplacian_features = self.model(self.get_lap_filtered_image(content_path))

        # style_features = fs

        # get target features
        # target_features = [exact_feature_distribution_matching(fs[i], style_features[i]) for i in range(len(style_features)-1)] # efdm
        target_features = []
        for cur, lap in zip(fs, laplacian_features):
            target_features.append(cur - lap)

        # for cur, style in zip(fs, style_features):
        #     c_mean, c_std = calc_mean_std(cur)
        #     s_mean, s_std = calc_mean_std(style)
        #     LOGGER.debug(f'c_mean={c_mean}, c_std={c_std}, s_mean={s_mean}, s_std={s_std}')
        #     target_features.append(((cur - c_mean)/c_std) * s_std + s_mean)
                    
        # target_features = [f + torch.ones_like(f) for f in fs]
        # target_features = [f + f/2 * torch.rand_like(f) for f in fs]


        #! 特征插值
        # for i in range(len(style_features)):
        #     style_features[i] = (style_features[i] + fs[i])/2

        # optimize
        # opt_vars = [torch.nn.Parameter(level_image) for level_image in output_pyramid]

        if self.config.optimizer == 'lbfgs':
            # optimizer = optim.LBFGS(opt_vars, lr=0.1)
            optimizer = optim.LBFGS([opt_img], lr=1)
        elif self.config.optimizer == 'adam':
            optimizer = optim.Adam([opt_img], lr=self.config.lr)
        else: 
            raise RuntimeError(f'{self.config.optimizer} is not supported')

        n_iter = [0]

        while n_iter[0] <= self.config.max_iter:
            def closure():
                optimizer.zero_grad()
                # opt_img = syn_lap_pyr(opt_vars)
                c_feats, o_feats = self.model(opt_img)

                # content_loss = torch.mean((c_feats - content_features) ** 2)
                loss = 0
                
                total_variance_loss = tv_loss(opt_img)

                # for cur, style in zip(s_feats, style_features):
                for cur, style in zip(o_feats, target_features):
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
                    out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                    self.optimize_images[str(n_iter[0]//self.config.show_iter)] = out_img
                return loss
            
            optimizer.step(closure)
        
        # save final image
        # opt_img = syn_lap_pyr(opt_vars)
        self.optimize_images['result'] = self.tensor2pil(opt_img)
        
    
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
    