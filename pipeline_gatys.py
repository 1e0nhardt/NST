from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from PIL import Image
from torch import optim
from torch.autograd import Variable
from torchvision import models, transforms
from torchvision.utils import save_image


from losses import gram_loss, tv_loss
from models.vgg import VGG19FeatureExtractor
from utils import check_folder, get_basename, CONSOLE, erase
import tyro

@dataclass
class GatysConfig:
    """Gatys NST Arguments"""

    output_dir: str = "./results/gatys"
    """output dir"""
    lambda_s: float = 100
    """style weight"""
    max_iter: int = 800
    """LBFGS optimize steps"""
    show_iter: int = 50
    """frequencies to show current loss"""
    standardize: bool = False
    """use ImageNet mean to standardization"""
    input_range: int = 1
    """scale number range after normalization and standardization"""
    use_in: bool = False
    """apply instance normalization on all style feature maps"""
    resize: int = 512
    """image resize"""
    verbose: bool = False
    """show additional information"""


class GatysPipeline(object):
    def __init__(self, config: GatysConfig) -> None:
        self.config = config
        CONSOLE.print(config)
        check_folder(self.config.output_dir)

        # prepare model
        self.vgg = VGG19FeatureExtractor(self.config.use_in)
        self.vgg.freeze_params()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # prepare transforms
        transform_pre_list = []
        transform_post_list = []
        transform_pre_list.append(transforms.Resize(self.config.resize))
        transform_pre_list.append(transforms.ToTensor()) # PIL to Tensor
        if self.config.standardize:
            transform_pre_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1,1,1]))
            transform_post_list.append(transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]))
        if self.config.input_range != 1:
            transform_pre_list.append(transforms.Lambda(lambda x: x.mul_(self.config.input_range)))
            transform_post_list.append(transforms.Lambda(lambda x: x.mul_(1/self.config.input_range)))
        transform_post_list.reverse()
        transform_post_list.append(transforms.Lambda(lambda x: torch.clamp(x, min=0, max=1)))
        transform_post_list.append(transforms.ToPILImage()) # Tensor to PIL

        self.transform_pre = transforms.Compose(transform_pre_list)
        self.transform_post = transforms.Compose(transform_post_list)
    
    def __call__(self, content_path: str, style_path: str):
        # prepare input tensors: (1, c, h, w), (1, c, h, w)
        content_image = self.transform_pre(Image.open(content_path)).unsqueeze(0).to(self.device)
        style_image = self.transform_pre(Image.open(style_path)).unsqueeze(0).to(self.device)
        if style_image.shape[1] == 1: # grayscale image
            style_image = repeat(style_image, 'b c h w -> b (repeat c) h w', repeat=3)

        # optimize target
        opt_img = content_image.data.clone()
        # opt_img = self.transform_pre(Image.open("./results/gatys/Tuebingen_Neckarfront_6_100_1_raw_raw_tv.png")).unsqueeze(0)

        # opt_img = erase(opt_img, 100, 100, 200, 200, content_image[..., 100:300, 100:300])
        # opt_img = transforms.RandomErasing(p=1)(opt_img)
        # opt_img = transforms.RandomErasing(p=1)(opt_img)
        # erase_mask = torch.rand_like(opt_img[0, 0]) > 0.4

        # erase_mask = torch.zeros_like(opt_img[0, 0])
        # erase_mask[100:300, 100:300] = 1
        # erase_mask = (erase_mask == 1)
        # erase_mask = erase_mask[None, None, ...]
        # self.erase_mask = repeat(erase_mask, 'b c h w -> b (r c) h w', r=3)

        # out_img = self.transform_post(erase_mask.detach().cpu().squeeze())
        # out_img.save(self.config.output_dir + f'/mask_erased.png')
        # exit()

        # opt_img[self.erase_mask] = 0
        opt_img = opt_img.to(self.device)
        opt_img.requires_grad = True

        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        out_img.save(self.config.output_dir + f'/{get_basename(content_path)}_{get_basename(style_path)}_{self.config.lambda_s}_{self.config.input_range}_{"std" if self.config.standardize else "raw"}_{"in" if self.config.use_in else "raw"}_erased.png')

        # get target features
        _, style_features = self.vgg.forward(style_image)
        content_features, fs = self.vgg.forward(content_image)
        if self.config.verbose: # 查看各层特征图的实际内容
            for i, feats in enumerate(fs):
                CONSOLE.print('_'*42)
                CONSOLE.print(feats.shape)
                CONSOLE.print(feats.min())
                CONSOLE.print(feats.max())
                CONSOLE.print(feats.mean())

                save_image(torch.clamp(repeat(feats.squeeze()[16:32], 'c h w -> c h w ch', ch=3).permute(0, 3, 1, 2), 0, 255), f'grid_{i+1}_1.png', nrow=4, normalize=True)
                if i == 3:
                    plt.hist(feats[0, 1].flatten().detach().cpu().numpy(), bins=255, range=(0, 256), log=True)
            plt.show()

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
                total_variance_loss = tv_loss(opt_img)
                loss = self.config.lambda_s * style_loss + content_loss + 1 * total_variance_loss
                loss.backward()
                # if n_iter[0] <= 400:
                #     opt_img.grad[~self.erase_mask] = 0 # 将不需要优化的区域的梯度置为零
                n_iter[0] += 1

                #print loss
                if n_iter[0] % self.config.show_iter == 0:
                    CONSOLE.print(f'Iteration: {n_iter[0]}, loss: {loss.item():.4f}, style_loss: {style_loss.item():.4f}, content_loss: {content_loss.item():.4f}')
                    out_img = self.transform_post(opt_img.detach().cpu().squeeze())
                    out_img.save(self.config.output_dir + f'/{get_basename(content_path)}_{get_basename(style_path)}_{self.config.lambda_s}_{self.config.input_range}_{"std" if self.config.standardize else "raw"}_{"in" if self.config.use_in else "raw"}_tv_inpaint_{n_iter[0]}.png')
                return loss
            
            optimizer.step(closure)

        out_img = self.transform_post(opt_img.detach().cpu().squeeze())
        plt.imshow(out_img)
        plt.show()
        out_img.save(self.config.output_dir + f'/{get_basename(content_path)}_{get_basename(style_path)}_{self.config.lambda_s}_{self.config.input_range}_{"std" if self.config.standardize else "raw"}_{"in" if self.config.use_in else "raw"}_tv_inpaint.png')


if __name__ == '__main__':
    args = tyro.cli(GatysConfig)
    pipeline = GatysPipeline(args)
    pipeline(
        'data/Tuebingen_Neckarfront.jpg', 
        'data/style/6.jpg'
    )
