import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import models, transforms

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, use_in=False) -> None:
        super().__init__()
        self.use_in = use_in
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    
    def forward(self, x):
        if self.use_in:
            relu1_1 = F.instance_norm(self.model[:2](x))*255
            relu2_1 = F.instance_norm(self.model[2:7](relu1_1))*255
            relu3_1 = F.instance_norm(self.model[7:12](relu2_1))*255
            relu4_1 = F.instance_norm(self.model[12:21](relu3_1))*255
            relu4_2 = self.model[21:23](relu4_1)
            relu5_1 = F.instance_norm(self.model[21:30](relu4_2))*255
        else:
            relu1_1 = self.model[:2](x)
            relu2_1 = self.model[2:7](relu1_1)
            relu3_1 = self.model[7:12](relu2_1)
            relu4_1 = self.model[12:21](relu3_1)
            relu4_2 = self.model[21:23](relu4_1)
            relu5_1 = self.model[21:30](relu4_2)
        return relu4_2, [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False