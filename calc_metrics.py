import os
import re
import warnings

import torch
from PIL import Image
from piq import LPIPS, ssim
from rich.progress import track
from torchvision import transforms

from losses import gram_loss
from models.vgg import VGG16FeatureExtractor
from utils import (LOGGER, AorB, ResizeHelper, ResizeMaxSide,
                         TimeRecorder, check_folder, get_basename, images2gif,
                         str2list)
from utils_st import calc_mean_std, covariance_loss, split_downsample

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # OUTPUT_DIR = 'results/TestMetrics/Test_AdaAttn/'
    # OUTPUT_DIR = 'results/TestMetrics/Test_ArtFlow-AdaIN/'
    # OUTPUT_DIR = 'results/TestMetrics/Test_ArtFlow-WCT/'
    # OUTPUT_DIR = 'results/TestMetrics/Test_StyleTr2/'
    # OUTPUT_DIR = 'results/TestMetrics/Test_WCT/'
    # OUTPUT_DIR = 'results/TestMetrics/Test_WCT_0.2/'
    # OUTPUT_DIR = 'results/TestMetrics/pama/'
    # OUTPUT_DIR = 'results/TestMetrics/Test_NNST/'
    # OUTPUT_DIR = 'results/TestMetrics/Test_STROTSS/'

    # OUTPUT_DIR = r'results/TestMetrics/nnst/common_25-18-11-6-1_remd/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/common_1-6-11-18-25_mean/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/common_1-6-11-18-25_std/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/common_1-6-11-18-25_adain/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/common_1-6-11-18-25_cov/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/common_1-6-11-18-25_efdm/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/common_1-6-11-18-25_gatysdivc/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_cov+remd2pyr/'
    OUTPUT_DIR = r'results/TestMetrics/gatys/common_1-6-11-18-25_cov_weights_1155/'
    # OUTPUT_DIR = r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs/'
    # OUTPUT_DIR = r'results/TestMetrics/nnfm/common_25-18-11-6-1_FS_nn'
    # OUTPUT_DIR = r'results/TestMetrics/nnfm/common_25-18-11-6-1_FS_remd'
    CONTENT_DIR = 'data/contents/'
    STYLE_DIR = 'data/styles/'
    device = 'cuda'

    model = VGG16FeatureExtractor(std=True)
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((512, 512), antialias=True),
        transforms.ToTensor()
    ])

    lpips = LPIPS()

    trecorder = TimeRecorder('Metrics')

    for output in track(sorted(os.listdir(OUTPUT_DIR))):
        if output.endswith('csv'):
            continue

        matches = re.findall(r'\d+', output)

        if OUTPUT_DIR == 'results/TestMetrics/Test_AdaAttn/':
            content = matches[1]
            style = matches[0]
        else:
            content = matches[0]
            style = matches[1]
        
        content_path = CONTENT_DIR + (f'{content}.png' if content in '1, 2, 6, 7, 8, 9' else f'{content}.jpg')
        style_path = STYLE_DIR + f'{style}.jpg'
        output_path = OUTPUT_DIR + output

        Ic = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        Is = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        Ics = transform(Image.open(output_path).convert("RGB")).unsqueeze(0).to(device)

        LOGGER.debug(output)
        LOGGER.debug(content_path)
        LOGGER.debug(style_path)

        ssim_metric = ssim(Ic, Ics)
        lpips_metric = lpips(Is, Ics)
        # lpips_metric = torch.tensor(-1)
        trecorder.reset_time()
        trecorder.set_record('Optimization', '-1')
        trecorder.set_record('SSIM', f'{ssim_metric.item():.4f}')
        trecorder.set_record('LPIPS', f'{lpips_metric.item():.4f}')

        mean_loss, std_loss, cov_loss, mean_loss_before, std_loss_before, cov_loss_before = 0, 0, 0, 0, 0, 0

        with torch.no_grad():
            Ics_41, Ics_featrues = model(Ics)
            Is_41, Is_featrues = model(Is)
            Ic_41, Ic_featrues = model(Ic)
            mean_diffs = []
            for cs, s, c in zip(Ics_featrues, Is_featrues, Ic_featrues):
                cs_mean, cs_std = calc_mean_std(cs)
                s_mean, s_std = calc_mean_std(s)
                c_mean, c_std = calc_mean_std(c)
                mean_loss_before += torch.mean((c_mean - s_mean)**2)
                std_loss_before += torch.mean((c_std - s_std)**2)
                cov_loss_before += covariance_loss(c, s)
                mean_loss += torch.mean((cs_mean - s_mean)**2)
                std_loss += torch.mean((cs_std - s_std)**2)
                cov_loss += covariance_loss(cs, s)
                mean_diff_ccs = (abs(c_mean - cs_mean)).squeeze().cpu()
                mean_diff_cs = (abs(c_mean - s_mean)).squeeze().cpu()
                mean_diffs.append((mean_diff_ccs, mean_diff_cs))

            # content_loss_before = torch.mean((Ic_41 - Is_41)**2)
            content_loss = torch.mean((Ics_41 - Ic_41)**2)
            g_loss = gram_loss(Ics_featrues, Is_featrues)
            
        trecorder.set_record('Mean Loss', f'{mean_loss:.3e}')
        trecorder.set_record('L_m Before', f'{mean_loss_before:.3e}')
        trecorder.set_record('Std Loss', f'{std_loss:.3e}')
        trecorder.set_record('L_std Before', f'{std_loss_before:.3e}')
        trecorder.set_record('Cov Loss', f'{cov_loss:.4f}')
        trecorder.set_record('L_cov Before', f'{cov_loss_before:.4f}')
        trecorder.set_record('Content Loss', f'{content_loss:.4f}')
        # trecorder.set_record('L_c Before', f'{content_loss_before:.4f}')
        trecorder.show()
        trecorder.save(OUTPUT_DIR, content + '_' + style)
        
