import math
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import models, transforms


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
    
    def forward(self, ix, hypercolumn=False, weight_factor=1):
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
        style_features = [f/math.pow(f.shape[1], weight_factor) for f in style_features]

        if hypercolumn:
            # resize and concat
            style_features = torch.cat([F.interpolate(f, (h // 4, w // 4), mode='bilinear', align_corners=True) for f in style_features], 1) 

        return content_features, style_features
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    from rich.console import Console
    CONSOLE = Console()
    from torch.utils.tensorboard import SummaryWriter
    from inception import InceptionFeatureExtractor

    # vgg16 = VGG16FeatureExtractor().cuda()
    # vgg16 = InceptionFeatureExtractor().cuda()
    # writer = SummaryWriter('log/image_downsample/test01')
    image = Image.open('data/content/C2.png')
    # image = Image.open('data/nnst_style/S4.jpg')
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Lambda(lambda x: x.mul_(2))
    ])
    transp = transforms.Compose([
        transforms.ToPILImage(),
    ])
    x = trans(image).cuda()
    def split_downsample(x, size):
        outputs = []
        for i in range(size):
            for j in range(size):
                outputs.append(x[:, i::size, j::size])
        return outputs
    outputs = split_downsample(x, 4)

    for i, output in enumerate(outputs):
        o = transp(output)
        o.save(f'results/image/output_{i}.png')

    # x = torch.ones([1, 3, 512, 512], device='cuda') * 2
    
    # _, sf = vgg16(x)

    # for i, f in enumerate(sf, 1):
    #     writer.add_images('feat_image', repeat(rearrange(f, 'n c h w -> c n h w'), 'n c h w -> n (m c) h w', m=3), i, dataformats='NCHW')

    # for p in vgg19.parameters():
    #     print(p.mean())
    # torch.manual_seed(42)
    # x = torch.rand([1,3,256,256], device='cuda') *100
    # x.requires_grad_(True)
    # vgg19.to('cuda')
    # c_feats, s_feats = vgg19(x)
    # loss = 1e7*sum([torch.mean((s - torch.ones_like(s)*1)**2) for s in s_feats])
    # loss.backward()
    # for feats in s_feats:
    #     CONSOLE.print(feats.mean().item(), feats.var().item(), feats.max().item(), feats.min().item())
    # CONSOLE.print(x.grad.mean().item(), x.grad.var().item(), x.grad.max().item(), x.grad.min().item(), style='red')
    # exit()
    # print(c_feats.shape)

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
