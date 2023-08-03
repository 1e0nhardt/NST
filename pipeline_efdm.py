from collections import OrderedDict
from dataclasses import dataclass

import torch
import tyro
from einops import rearrange, repeat
from PIL import Image
from torch import optim

from utils import LOGGER, check_folder, AorB
from utils_st import exact_feature_distribution_matching
from losses import gram_loss
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
    name: str = 'efdm'
    """pipeline name."""
    lambda_s: float = 1
    """style weight"""
    use_tv_loss: bool = False
    """use total variance loss"""
    model_type: str = 'vgg16'
    """feature extractor model type vgg16 | vgg19 | inception"""
    # layers: str = "1,6,11,20,29"
    # layers: str = "1"
    layers: str = "1,6,11,18,25"
    """layer indices of style features which should seperate by ','"""
    optimizer: str = 'lbfgs'
    """optimizer type: adam | lbfgs"""
    use_wandb: bool = False
    use_tensorboard: bool = True
    """use tensorboard"""
    max_iter: int = 500
    verbose: bool = False


class EFDMPipeline(OptimzeBasePipeline):
    def __init__(self, config: EFDMConfig) -> None:
        super().__init__(config)
        self.config = config
        check_folder(self.config.output_dir)
    
    def add_extra_infos(self):
        infos = [self.config.layers]
        infos += [str(self.config.lambda_s)]
        infos += ['gramreg']
        return infos
    
    def optimize_process(self, content_path, style_path, mask=False):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        # convert("RGB")会让灰度图像也变为3通道
        style_image = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)
        # img = content_image
        # content_image = style_image
        # style_image = img

        # optimize target
        optimize_images = OrderedDict()
        # opt_img = torch.rand_like(content_image)
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
        optimize_images['init'] = out_img

        opt_img.requires_grad = True

        # get target features
        _, style_features = self.model.forward(style_image)
        # for i, f in enumerate(style_features, 1):
        #     LOGGER.warning(f.max())
        #     LOGGER.warning(f.mean())
        #     self.writer.add_images('feat_image', repeat(rearrange(f, 'n c h w -> c n h w'), 'n c h w -> n (m c) h w', m=3), global_step=i, dataformats='NCHW')
        content_features, fs = self.model.forward(content_image)
        # for i, f in enumerate(fs, 1):
        #     LOGGER.warning(f.max())
        #     LOGGER.warning(f.mean())
        #     self.writer.add_images('content_image', repeat(rearrange(f, 'n c h w -> c n h w'), 'n c h w -> n (m c) h w', m=3), global_step=i, dataformats='NCHW')
        #! 直接用内容图的风格图的特征进行EFDM匹配得到的特征作为目标来优化
        #! 结果有很多噪点，内容结构保存较好，颜色更偏风格图。
        target_features = [exact_feature_distribution_matching(fs[i], style_features[i]) for i in range(len(style_features))] # 4-1

        for cur in style_features:
            #* cur.shape[1] 是补偿vgg16中的分层加权操作
            cur *= cur.shape[1]

        for cur in target_features:
            #* cur.shape[1] 是补偿vgg16中的分层加权操作
            cur *= cur.shape[1]

        # optimize
        # optimizer = optim.Adam([opt_img], lr=1e-2)
        optimizer = optim.LBFGS([opt_img], lr=1)
        n_iter = [0]

        while n_iter[0] <= self.config.max_iter:
            def closure():
                optimizer.zero_grad()

                _, s_feats = self.model(opt_img)

                for cur in s_feats:
                    #* cur.shape[1] 是补偿vgg16中的分层加权操作
                    cur *= cur.shape[1]

                style_loss = 0
                for i in range(len(s_feats)):
                    # style_loss += 10 * torch.mean((s_feats[i] - target_features[i]) ** 2)

                    #! 优化时不需要先得到目标特征
                    #! EFDM算法就是用内容图的序，排列风格图的特征。
                    input = s_feats[i]
                    target = style_features[i]
                    B, C, W, H = input.size(0), input.size(1), input.size(2), input.size(3)
                    _, index_content = torch.sort(input.view(B, C, -1))
                    value_style, _ = torch.sort(target.view(B, C, -1))
                    inverse_index = index_content.argsort(-1)
                    style_loss += torch.mean((input.view(B,C,-1)-value_style.gather(-1, inverse_index))**2)


                # total_variance_loss = tv_loss(opt_img)
                cov_loss = gram_loss(s_feats, style_features)

                # loss = self.config.lambda_s * style_loss + content_loss #+ 1 * total_variance_loss
                loss = style_loss + cov_loss# + content_loss #+ 1 * total_variance_loss
 
                loss.backward()
                # 将不需要优化的区域的梯度置为零
                if mask:
                    opt_img.grad[~self.erase_mask] = 0
                n_iter[0] += 1

                #print loss
                if n_iter[0] % self.config.show_iter == 0:
                    LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}')
                    # LOGGER.info(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                    out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                    optimize_images[str(n_iter[0]//self.config.show_iter)] = out_img
                return loss
            
            optimizer.step(closure)
        
        # save final image
        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        optimize_images['result'] = out_img
        
        return optimize_images


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

    for s in os.listdir(style_dir):
        pipeline(
            'data/content/sailboat.jpg',
            style_dir + s
        )

    # pipeline(
    #     'data/content/sailboat.jpg', 
    #     'data/style/130.jpg',
    #     'data/nnst_style/shape.png',
    #     mask=False
    # )
