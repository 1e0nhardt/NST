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
    def __init__(self) -> None:
        super().__init__()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    
    def forward(self, x, hypercolumn=False):
        style_features = []
        content_features = None
        h, w = x.shape[2:]

        # N x 3 x 299 x 299
        x = self.model.Conv2d_1a_3x3(x)
        # CONSOLE.print(x.mean(), x.var())
        style_features.append(x)
        # N x 32 x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 32 x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 64 x 147 x 147
        x = self.model.maxpool1(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 64 x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # CONSOLE.print(x.mean(), x.var())
        style_features.append(x)
        # N x 80 x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 192 x 71 x 71
        x = self.model.maxpool2(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 192 x 35 x 35
        x = self.model.Mixed_5b(x)
        # CONSOLE.print(x.mean(), x.var())
        style_features.append(x)
        # N x 256 x 35 x 35
        x = self.model.Mixed_5c(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 288 x 35 x 35
        x = self.model.Mixed_5d(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 288 x 35 x 35
        x = self.model.Mixed_6a(x)
        # CONSOLE.print(x.mean(), x.var())
        style_features.append(x)
        content_features = x.clone()
        # N x 768 x 17 x 17
        x = self.model.Mixed_6b(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 768 x 17 x 17
        x = self.model.Mixed_6c(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 768 x 17 x 17
        x = self.model.Mixed_6d(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 768 x 17 x 17
        x = self.model.Mixed_6e(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 768 x 17 x 17
        x = self.model.Mixed_7a(x)
        # CONSOLE.print(x.mean(), x.var())
        style_features.append(x)
        # N x 1280 x 8 x 8
        x = self.model.Mixed_7b(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 2048 x 8 x 8
        x = self.model.Mixed_7c(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.model.avgpool(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 2048 x 1 x 1
        x = self.model.dropout(x)
        # CONSOLE.print(x.mean(), x.var())
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.model.fc(x)
        # CONSOLE.print(x.mean(), x.var())

        if hypercolumn:
            # Normalize each layer by # channels so # of channels doesn't dominate 
            style_features = [f/f.shape[1] for f in style_features]
            # resize and concat
            style_features = torch.concat([F.interpolate(f, (h // 4, w // 4), mode='bilinear', align_corners=True) for f in style_features], 1) 

        return content_features, style_features
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    inception = InceptionFeatureExtractor().cuda()
    inception.eval()
    # for name, layer in inception.model.named_parameters():
    #     print(name, layer.shape, layer.mean())
    torch.manual_seed(42)

    x = torch.rand([1,3,256,256], device='cuda', requires_grad=True)
    c, s_feats = inception(x)

    loss = sum([torch.mean((s - torch.ones_like(s)*0.9)**2) for s in s_feats])
    loss.backward()

    for feats in s_feats:
        CONSOLE.print(feats.mean().item(), feats.var().item(), feats.max().item(), feats.min().item())
    CONSOLE.print(x.grad.mean().item(), x.grad.var().item(), x.grad.max().item(), x.grad.min().item(), style='red')
    
    # print(c.shape)
    # for s in ss:
    #     print(s.shape)