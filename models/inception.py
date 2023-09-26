import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from rich.console import Console

CONSOLE = Console()

class InceptionFeatureExtractor(nn.Module):
    """
    Layers used:
        conv2d_1a \n
        conv2d_3b \n
        mixed_5b \n
        mixed_6  \n
        mixed_7a \n
    """
    def __init__(self, std) -> None:
        super().__init__()
        self.std = std
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    
    def forward(self, x, hypercolumn=False, normalize=False):
        style_features = []
        content_features = None
        h, w = x.shape[2:]

        # N x 3 x 299 x 299
        x = self.model.Conv2d_1a_3x3(x)
        style_features.append(x)
        # N x 32 x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        style_features.append(x)
        # N x 80 x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.model.Mixed_5b(x)
        style_features.append(x)
        # N x 256 x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_6a(x)
        style_features.append(x)
        content_features = x.clone()
        # N x 768 x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_7a(x)
        style_features.append(x)
        # N x 1280 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.model.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.model.fc(x)

        if normalize:
            # Normalize each layer by # channels so # of channels doesn't dominate 
            style_features = [f/f.shape[1] for f in style_features]

        if hypercolumn:
            # resize and concat
            style_features = torch.concat([F.interpolate(f, (h // 4, w // 4), mode='bilinear', align_corners=True) for f in style_features], 1) 

        return content_features, style_features
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False