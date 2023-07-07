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
    def __init__(self, layers=[1, 6, 11, 20, 29], use_in=False) -> None:
        super().__init__()
        self.use_in = use_in
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = layers
    
    def forward(self, x):
        final_ix = max(self.layers)
        style_features = []
        content_features = None

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers:
                if self.use_in:
                    x = F.instance_norm(x)
                style_features.append(x)
            elif i == 22:
                content_features = x

            if i == final_ix:
                break

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
    def __init__(self, layers=[1, 6, 11, 18, 25], use_in=False) -> None:
        super().__init__()
        self.use_in = use_in
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers
    
    def forward(self, x):
        final_ix = max(self.layers)
        style_features = []
        content_features = None

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers:
                if self.use_in:
                    x = F.instance_norm(x)
                style_features.append(x)
            elif i == 20:
                content_features = x

            if i == final_ix:
                break

        return content_features, style_features
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    from rich.console import Console
    CONSOLE = Console()

    vgg19 = VGG19FeatureExtractor()
    for i, layer in enumerate(vgg19.model):
        if 'MaxPool' in str(layer):
            CONSOLE.print(i, '-->', str(layer))
        else:
            CONSOLE.print(i, str(layer))
    
    CONSOLE.print('*'*80)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    for i, layer in enumerate(vgg16):
        if 'MaxPool' in str(layer):
            CONSOLE.print(i, '-->', str(layer))
        else:
            CONSOLE.print(i, str(layer))