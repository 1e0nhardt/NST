import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torchvision import models, transforms

sys.path.append('D:/MyCodes/NST')

from utils_st import calc_mean_std


class VGG19FeatureExtractor(nn.Module):
    """
    Layer indexes:
        Conv1_*: 1,3 \n
        Conv2_*: 6,8 \n
        Conv3_*: 11, 13, 15, 17 \n
        Conv4_*: 20, 22, 24, 26 \n
        Conv5_*: 29, 31, 33, 35 \n
    """
    def __init__(self, layers=[1, 6, 11, 20, 29], std=False) -> None:
        super().__init__()
        self.std = std
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = layers
    
    def forward(self, ix, hypercolumn=False):
        x = ix.clone() # 标准化时不修改原output image的数值
        final_ix = max(self.layers)
        style_features = []
        content_features = None
        h, w = x.shape[2:]

        if self.std:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            for i in range(3):
                x[:, i:(i + 1), :, :] = (x[:, i:(i + 1), :, :] - mean[i]) / std[i]

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers:
                style_features.append(x)
            
            if i == 22:
                content_features = x.clone()

            if i == final_ix:
                break
            
        if hypercolumn:
            # Normalize each layer by # channels so # of channels doesn't dominate 
            style_features = [f/f.shape[1] for f in style_features]
            # resize and concat
            style_features = torch.cat([F.interpolate(f, (h // 4, w // 4), mode='bilinear', align_corners=True) for f in style_features], 1) 

        return content_features, style_features
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False


class VGG16FeatureExtractor(nn.Module):
    """
    Layer indexes:
        Conv1_*: 1,3 \n   
        Conv2_*: 6,8 \n
        Conv3_*: 11, 13, 15 \n
        Conv4_*: 18, 20, 22 \n
        Conv5_*: 25, 27, 29 \n
    """
    def __init__(self, layers=[1, 6, 11, 18, 25], std=False) -> None:
        super().__init__()
        self.std = std
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers
    
    def forward(self, ix, hypercolumn=False, weight_factor=1, normalize=False):
        x = ix.clone() # 标准化时不修改原output image的数值
        final_ix = max(self.layers)
        style_features = []
        content_features = None
        h, w = x.shape[2:]

        if self.std:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            for i in range(3):
                x[:, i:(i + 1), :, :] = (x[:, i:(i + 1), :, :] - mean[i]) / std[i]

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers:
                style_features.append(x)

            if i == 20:
                content_features = x.clone()

            if i == final_ix:
                break
        
        #! Normalize each layer by # channels so # of channels doesn't dominate 
        if normalize:
            style_features = [f / math.pow(f.shape[1], weight_factor) for f in style_features]

        if hypercolumn:
            # resize and concat
            style_features = torch.cat([F.interpolate(f, (h // 4, w // 4), mode='bilinear', align_corners=True) for f in style_features], 1) 

        return content_features, style_features
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    from rich.console import Console
    from rich.progress import track
    CONSOLE = Console()
    from inception import InceptionFeatureExtractor
    from torch.utils.tensorboard import SummaryWriter

    vgg16 = VGG16FeatureExtractor(layers=[25,18,11,6,1], std=True).cuda()
    # vgg16 = InceptionFeatureExtractor().cuda()
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    # print(vgg16[:3])
    for i, layer in enumerate(vgg16):
        if 'MaxPool' in str(layer):
            CONSOLE.print(i, '-->', str(layer))
        else:
            CONSOLE.print(i, str(layer))

    # for i, layer in enumerate(vgg16.new_model):
    #     CONSOLE.print(i, str(layer))

    exit()
    writer = SummaryWriter('log/vgg16/features/c1s4adaintarget')
    image_c = Image.open('data/content/C1.png')
    # image = Image.open('data/style/130.jpg')
    image_s = Image.open('data/nnst_style/S4.jpg')
    image_cs = Image.open('results/efdm/vgg19_adam_std_R1_100/C2_S1_0.001_result.png')
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Lambda(lambda x: x.mul_(2))
    ])
    transp = transforms.Compose([
        transforms.ToPILImage(),
    ])
    c = trans(image_c).cuda().unsqueeze(0)
    s = trans(image_s).cuda().unsqueeze(0)
    cs = trans(image_cs).cuda().unsqueeze(0)
    alpha = 0.1
    x = ((1-alpha)*c+alpha*s)/2
    x = cs


    # x = torch.ones([1, 3, 512, 512], device='cuda') * 2
    
    _, cf = vgg16(c, normalize=False)
    _, sf = vgg16(s, normalize=False)

    # for i, f in track(enumerate(sf, 1)):
    #     writer.add_images('feat_image', repeat(rearrange(f, 'n c h w -> c n h w'), 'n c h w -> n (m c) h w', m=3), i, dataformats='NCHW')
    
    i = 0
    for cur, style in zip(cf, sf):
        c_mean, c_std = calc_mean_std(cur)
        s_mean, s_std = calc_mean_std(style)
        target = ((cur - c_mean)/c_std*s_std + s_std)
        writer.add_images('feat_image', repeat(rearrange(target, 'n c h w -> c n h w'), 'n c h w -> n (m c) h w', m=3), i, dataformats='NCHW')
        i+=1

    # for i, layer in enumerate(vgg19.model):
    #     if 'MaxPool' in str(layer):
    #         CONSOLE.print(i, '-->', str(layer))
    #     else:
    #         CONSOLE.print(i, str(layer))
    
    # CONSOLE.print('*'*80)

    # vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    # for i, layer in enumerate(vgg16):
    #     if 'MaxPool' in str(layer):
    #         CONSOLE.print(i, '-->', str(layer))
    #     else:
    #         CONSOLE.print(i, str(layer))
