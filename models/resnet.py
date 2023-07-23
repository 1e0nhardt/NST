from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import models, transforms


class Resnet50FeatureExtractor(nn.Module):
    """
    Layer indexes:
        Conv1_*: 1,3 \n   
        Conv2_*: 6,8 \n
        Conv3_*: 11, 13, 15 \n
        Conv4_*: 18, 20, 22 \n
        Conv5_*: 25, 27, 29 \n
    """
    def __init__(self) -> None:
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    def forward(self, x, hypercolumn=False):
        style_features = []
        content_features = None
        h, w = x.shape[2:]

        if False:
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

        if hypercolumn:
            # Normalize each layer by # channels so # of channels doesn't dominate 
            style_features = [f/f.shape[1] for f in style_features]
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

    def softmax(feats, T = 1000):
        _, _, h, w = feats.shape
        x = rearrange(feats, 'n c h w -> n c (h w)')
        x = x / T
        x = torch.softmax(x, dim=-1)
        return rearrange(x, 'n c (h w) -> n c h w', h=h, w=w)

    model = Resnet50FeatureExtractor().cuda()
    writer = SummaryWriter('log/test10')
    image = Image.open('data/content/C2.png')
    # image = Image.open('data/nnst_style/S7.jpg')
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = trans(image).unsqueeze(0).cuda()

    x = torch.ones([1, 3, 512, 512], device='cuda')
    
    _, sf = model(x)

    i = 0
    for f in sf:
        i += 1
        print(f.squeeze().shape)
        # f = softmax(f)
        for i in range(f.shape[1]):
            mi = f[:, i, :, :].min()
            me = f[:, i, :, :].mean()
            mx = f[:, i, :, :].max()
            print(me, ', ', mx)
            f[:, i, :, :] = (f[:, i, :, :] - mi + 1e-10)/(mx-mi)
        writer.add_images('feat_image', repeat(rearrange(f, 'n c h w -> c n h w'), 'n c h w -> n (m c) h w', m=3), i, dataformats='NCHW')


    # res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # res.named_modules()

    # for i, layer in res.named_modules():
    #     if 'MaxPool' in str(layer):
    #         CONSOLE.print(i, '-->', str(layer))
    #     else:
    #         CONSOLE.print(i, str(layer))
