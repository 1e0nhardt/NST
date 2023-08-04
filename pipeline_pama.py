from dataclasses import dataclass
import torch
from pama.pama_model import PAMANet
from utils import ResizeHelper
from PIL import Image
from torchvision.transforms import transforms
from utils import tensor2img, showimgs, save_image
import os


@dataclass
class PamaConfig:
    """Pama Model Arguments"""

    checkpoints: str = "D:/MyCodes/nerfstudio/pretrained/pama/original"
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
    from torchvision.utils import make_grid
    from torchvision.io import read_image
    from pathlib import PurePath
    import tyro

    pama_config = tyro.cli(PamaConfig)
    helper = ResizeHelper()

    # content_path = "data/content/frame_00001.png"
    style_path = "data/nnst_style/S1.jpg"
    # style_path = "data/style/17.jpg"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.RandomCrop((256, 256)),
        transforms.RandomEqualize(p=1),
        transforms.ToTensor(), # [0, 1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
        transforms.Lambda(lambda x: x.mul_(pama_config.input_range))
    ])
    
    transform2 = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.RandomCrop((256, 256)),
        # transforms.RandomEqualize(p=1),
        transforms.ToTensor(), # [0, 1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
        transforms.Lambda(lambda x: x.mul_(pama_config.input_range))
    ])
    

    Is = transform2(Image.open(style_path).convert("RGB")).unsqueeze(0)
    st_images = []
    for content_path in os.listdir('data/content'):
        Ic = transform(Image.open(f'data/content/{content_path}').convert("RGB")).unsqueeze(0)
        # Ic = torch.rand_like(Ic)
        # Ic = 0.8*Ic + 0.2*torch.rand_like(Ic)
        name = os.path.basename(content_path)

        padded_image = helper.pad_to8x(Ic)
        style_transferred_image = pama_infer_one_image(padded_image, Is, pama_config)
        #! resize to original image size
        style_transferred_image = helper.crop_to_original(style_transferred_image)

        # ret = tensor2img(style_transferred_image)
        # showimgs(1, 1, [ret], ['pama'])
        st_images.append(style_transferred_image*1.5)

        # save_image(style_transferred_image, f'results/lego/{name}_14.jpg')
    
    grid = make_grid(torch.cat(st_images), nrow=4, pad_value=1)
    save_image(grid, f'grid_{pama_config.input_range}_S1_0.png')




