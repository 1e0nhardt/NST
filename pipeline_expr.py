from collections import OrderedDict
from dataclasses import dataclass

import torch
import tyro
from einops import rearrange, repeat
from PIL import Image
from torch import optim

from losses import gram_loss, tv_loss
from nnst.image_pyramid import dec_lap_pyr, syn_lap_pyr
from utils import LOGGER, AorB, check_folder
from utils_st import exact_feature_distribution_matching, softmax_smoothing, get_gaussian_conv, softmax_smoothing_2d
from pipeline_optimize import OptimzeBaseConfig, OptimzeBasePipeline
import math

@dataclass
class ExprConfig(OptimzeBaseConfig):

    name: str = 'explore'
    output_dir: str = "./results/explore"
    """output dir"""
    model_type: str = 'vgg16'
    """feature extractor model type vgg16 | vgg19 | inception"""
    # layers: str = "25"
    layers: str = "1,6,11,18,25"
    # layers: str = "25, 27, 29, 22, 20, 18, 15, 13, 11, 8, 6, 3, 1"
    """layer indices of style features which should seperate by ','"""
    use_wandb: bool = False
    use_tensorboard: bool = False
    optimizer: str = 'lbfgs'
    """optimizer type: adam | lbfgs"""
    verbose: bool = False
    max_iter: int = 500
    save_process: bool = True
    use_tv_loss: bool = False


class ExprPipeline(OptimzeBasePipeline):
    def __init__(self, config: ExprConfig) -> None:
        super().__init__(config)
        self.config = config
        check_folder(self.config.output_dir)
    
    def add_extra_infos(self):
        return [f'reverse_content_{self.config.layers}'] + AorB(self.config.use_tv_loss, '100tvloss')
    
    def add_extra_file_infos(self):
        return ['sinit']
    
    def optimize_process(self, content_path, style_path, mask=False):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        # convert("RGB")会让灰度图像也变为3通道
        style_image = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)

        # optimize target
        optimize_images = OrderedDict()

        # style_image = (content_image.clone() + style_image.clone())/2
        # opt_img = content_image.data.clone()
        opt_img = style_image.data.clone()
        # opt_img = torch.rand_like(content_image)
        opt_img = opt_img.to(self.device)

        # output_pyramid = dec_lap_pyr(content_image.clone(), 8)

        # save init image
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images['init'] = out_img

        opt_img.requires_grad = True

        # get target features
        _, style_features = self.model.forward(style_image)
        content_features, fs = self.model.forward(content_image)

        # style_features = fs

        # get target features
        # target_features = [exact_feature_distribution_matching(fs[i], style_features[i]) for i in range(len(style_features)-1)] # efdm
        target_features = fs


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
        for cur in target_features:
            #* cur.shape[1] 是补偿vgg16中的分层加权操作
            cur *= cur.shape[1]

        while n_iter[0] <= self.config.max_iter:
            def closure():
                optimizer.zero_grad()
                # opt_img = syn_lap_pyr(opt_vars)
                c_feats, s_feats = self.model(opt_img)

                # content_loss = torch.mean((c_feats - content_features) ** 2)
                loss = 0
                for cur in s_feats:
                    #* cur.shape[1] 是补偿vgg16中的分层加权操作
                    cur *= cur.shape[1]
                
                total_variance_loss = tv_loss(opt_img)

                # for cur, style in zip(s_feats, style_features):
                for cur, style in zip(s_feats, target_features):
                    loss += torch.mean((cur - style) ** 2)
                    # 交叉熵
                    # style = softmax_smoothing_2d(torch.sigmoid(style))
                    # cur = softmax_smoothing_2d(torch.sigmoid(cur))
                    # entropy_reg_loss = 1e10 * (style * (torch.log(style) - torch.log(cur))).mean()
                    # loss += entropy_reg_loss
                
                if self.config.use_tv_loss:
                    loss += 100 * total_variance_loss

                loss.backward()

                n_iter[0] += 1
                #print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')
                    out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                    optimize_images[str(n_iter[0]//self.config.show_iter)] = out_img
                return loss
            
            optimizer.step(closure)
        
        # save final image
        # opt_img = syn_lap_pyr(opt_vars)
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images['result'] = out_img
        
        return optimize_images

    
if __name__ == '__main__':
    args = tyro.cli(ExprConfig)
    pipeline = ExprPipeline(args)
    pipeline(
        'data/content/sailboat.jpg', 
        'data/nnst_style/S4.jpg',
        mask=False
    )
    