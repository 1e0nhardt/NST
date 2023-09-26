import os
from dataclasses import dataclass

import torch
from PIL import Image
from rich.progress import track
from torchvision.transforms import transforms

from pama.pama_model import PAMANet
from utils import ResizeHelper, get_basename, save_image


@dataclass
class PamaConfig:
    """Pama Model Arguments"""

    checkpoints: str = "data/pretrained/pama/original"
    """pretrained model path. Include encoder.pth, decoder.pth, and 3 PAMA*.pth"""
    pretrained: bool = True
    """use pretrained model"""
    requires_grad: bool = False
    """whether to finetune"""
    training: bool = False
    """we only need infer, so always set this to False"""
    input_range: float = 1


class PamaPipeline(object):
    def __init__(self, config: PamaConfig) -> None:
        self.config = config

def pama_infer_one_image(Ic, Is, hparams):
    model = PAMANet(hparams)
    model.eval()
    model.to('cuda')
    Ic = Ic.to('cuda')
    Is = Is.to('cuda')
    
    with torch.no_grad():
        Ics = model(Ic, Is)

    return Ics


if __name__ == '__main__':
    import tyro

    pama_config = tyro.cli(PamaConfig)
    helper = ResizeHelper()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(), # [0, 1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
        transforms.Lambda(lambda x: x.mul_(pama_config.input_range))
    ])
    
    transform2 = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(), # [0, 1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
        transforms.Lambda(lambda x: x.mul_(pama_config.input_range))
    ])
    

    import time
    start = time.time()
    st_images = []
    for content_path in track(os.listdir('data/contents')):
        for style_path in os.listdir('data/styles'):
            Ic = transform(Image.open(f'data/contents/{content_path}').convert("RGB")).unsqueeze(0)
            Is = transform2(Image.open(f'data/styles/{style_path}').convert("RGB")).unsqueeze(0)

            padded_image = helper.pad_to8x(Ic)
            style_transferred_image = pama_infer_one_image(padded_image, Is, pama_config)
            #! resize to original image size
            style_transferred_image = helper.crop_to_original(style_transferred_image)

            save_image(style_transferred_image, f'results/TestMetrics/pama_time/{get_basename(content_path)}_{get_basename(style_path)}.jpg')
    
    print((time.time() - start) / 256)




