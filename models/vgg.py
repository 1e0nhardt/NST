import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
            
            if i == 22:
                content_features = x.clone()

            if i == final_ix:
                break
        
        if normalize:
            style_features = [f / math.pow(f.shape[1], weight_factor) for f in style_features]
            
        if hypercolumn:
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