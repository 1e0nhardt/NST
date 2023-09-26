import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Resnet50FeatureExtractor(nn.Module):
    """
    Layer indexes:
        Conv1_*: 1,3 \n   
        Conv2_*: 6,8 \n
        Conv3_*: 11, 13, 15 \n
        Conv4_*: 18, 20, 22 \n
        Conv5_*: 25, 27, 29 \n
    """
    def __init__(self, std=False) -> None:
        super().__init__()
        self.std = std
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    def forward(self, x, hypercolumn=False, normalize=False):
        style_features = []
        content_features = None
        h, w = x.shape[2:]

        if self.std:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            for i in range(3):
                x[:, i:(i + 1), :, :] = (x[:, i:(i + 1), :, :] - mean[i]) / std[i]

        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        style_features.append(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        style_features.append(x)
        x = self.model.layer2(x)
        style_features.append(x)
        x = self.model.layer3(x)
        style_features.append(x)
        content_features = x
        x = self.model.layer4(x)
        style_features.append(x)

        if normalize:
            # Normalize each layer by # channels so # of channels doesn't dominate 
            style_features = [f/f.shape[1] for f in style_features]

        if hypercolumn:
            # resize and concat
            style_features = torch.cat([F.interpolate(f, (h // 4, w // 4), mode='bilinear', align_corners=True) for f in style_features], 1) 

        return content_features, style_features
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False
