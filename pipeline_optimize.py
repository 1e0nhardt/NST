from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torch import optim
from torchvision import models, transforms
from torchvision.utils import save_image

from losses import gram_loss, tv_loss
from models.vgg import VGG16FeatureExtractor, VGG19FeatureExtractor
from utils import CONSOLE, AorB, check_folder, get_basename, images2gif, str2list


@dataclass
class OptimzeBaseConfig:
    """
    基本训练配置
    """
    output_dir: str = "./results"
    """output dir"""
    max_iter: int = 500
    """optimize steps"""
    show_iter: int = 50
    """frequencies to show current loss"""
    use_vgg19: bool = True
    """use vgg19 as base model if True, otherwise vgg16"""
    layers: str = "1,6,11,20,29"
    """layer indices of style features which should seperate by ','"""
    standardize: bool = True
    """use ImageNet mean to standardization"""
    input_range: float = 1
    """scale number range after normalization and standardization"""
    use_in: bool = False
    """apply instance normalization on all style feature maps"""
    resize: int = 512
    """image resize"""
    save_process: bool = False
    """save images of optimize process as gif"""
    verbose: bool = False
    """show additional information"""


class OptimzeBasePipeline(object):
    def __init__(self, config: OptimzeBaseConfig) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config

        # prepare model
        self.vgg = self.prepare_vgg()
        
        # prepare transforms
        self.transform_pre, self.transform_post = self.prepare_transforms()
    
    def prepare_vgg(self):
        if self.config.use_vgg19:
            vgg = VGG19FeatureExtractor(str2list(self.config.layers), self.config.use_in)
        else:
            vgg = VGG16FeatureExtractor(str2list(self.config.layers), self.config.use_in)
        vgg.freeze_params()
        vgg.to(self.device)
        return vgg
    
    def prepare_transforms(self):
        transform_pre_list = []
        transform_post_list = []
        transform_pre_list.append(transforms.Resize((self.config.resize, self.config.resize)))
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

        return transforms.Compose(transform_pre_list), transforms.Compose(transform_post_list)
    
    def generate_expr_name(self, mask=False):
        infos = []
        infos += AorB(self.config.use_vgg19, 'vgg19', 'vgg16')
        infos += AorB(self.config.standardize, 'std', 'raw')
        infos += [str(self.config.input_range)]
        infos += AorB(self.config.use_in, 'IN')
        infos += self.add_extra_infos()
        infos += AorB(mask, 'mask')

        return '_'.join(infos)
    
    def add_extra_infos(self):
        return []
    
    def generate_filename(self, content_path: str, style_path:str):
        return get_basename(content_path) + '_' + get_basename(style_path)
    
    def vis_feature_activations(self, fs):
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
    
    def optimize_process(self, content_path, style_path, mask=False):
        assert False, "you should implement this method in extended class"

    def __call__(self, content_path: str, style_path: str, mask=False):
        expr_name = self.generate_expr_name(mask=mask)
        expr_dir = self.config.output_dir + '/' + expr_name
        check_folder(expr_dir)


        optimize_images = self.optimize_process(content_path, style_path, mask)
        
        filename = self.generate_filename(content_path, style_path)
        if self.config.save_process:
            images2gif(optimize_images, expr_dir + f'/{filename}_process.gif')
            optimize_images[0].save(expr_dir + f'/{filename}_init.png')
        optimize_images[-1].save(expr_dir + f'/{filename}_result.png')