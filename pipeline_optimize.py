import random
import time
import typing
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import wraps

import cv2
import numpy as np
import torch
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from PIL import Image
from piq import LPIPS, ssim
from rich.traceback import install
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.inception import InceptionFeatureExtractor
from models.vgg import VGG16FeatureExtractor, VGG19FeatureExtractor
from utils import (LOGGER, AorB, ResizeHelper, ResizeMaxSide,
                         TimeRecorder, check_folder, get_basename, images2gif,
                         str2list)
from utils_st import calc_mean_std, covariance_loss, split_downsample

warnings.filterwarnings('ignore')
install(show_locals=False) # 设置rich为默认的异常输出处理程序


@dataclass
class OptimzeBaseConfig:
    """
    基本训练配置
    """

    name: str = 'base'
    """expr name. Change this to your pipeline name."""
    output_dir: str = "./results/TTTT"
    """output dir"""
    max_iter: int = 500
    """optimize steps"""
    show_iter: int = 50
    """frequencies to show current loss"""
    save_init: bool = False
    """save init image"""
    save_process_gif: bool = False
    """save images of optimize process as gif"""
    save_rgbhist: bool = False
    """save rgb histogram of final result"""
    verbose: bool = False
    """show additional information"""
    use_wandb: bool = False
    """use wandb"""
    use_tensorboard: bool = False
    """use tensorboard"""
    show_metrics: bool = True
    """calculate and show ssim and lpips metrics"""
    content_path: str = 'data/contents/5.jpg'
    """content image path"""
    style_path: str = 'data/styles/12.jpg'
    """style image path"""

    model_type: str = 'vgg16'
    """feature extractor model type vgg16 | vgg19 | inception. Generally, vgg16 is enough."""
    layers: str = "1,6,11,18,25"
    """layer indices of style features which should seperate by ','"""
    optimizer: str = 'adam'
    """optimizer type: adam | lbfgs"""
    lr: float = 1e-3
    """learning rate"""
    optimize_strategy: str = 'common'
    """common: optimize rgb parameters, default; pyramid: optimize laplacian pyramid; cascade: optimze rgb but from coarse to fine"""
    standardize: bool = True
    """use ImageNet mean to standardization"""
    input_range: float = 1
    """scale number range. default 1."""
    max_side: int = 512
    """max side length of resized image"""
    exact_resize: bool = False
    """resize use exact size"""
    exact_size: tuple = (512, 512)
    """exact size of inputs"""
    use_tv_reg: bool = False
    """use total variance regularizer"""
    max_scales: int = 4
    """do ST on how many image pyramid levels"""
    pyramid_levels: int = 8
    """levels of image pyramid"""
    is_demo: bool = False


class OptimzeBasePipeline(object):
    def __init__(self, config: OptimzeBaseConfig) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        LOGGER.info(config)

        # prepare model
        self.model = self.prepare_model()
        
        # prepare transforms
        self.transform_pre, self.transform_post = self.prepare_transforms()

        self.lpips = LPIPS()

        # loggers
        self.time_record = TimeRecorder(self.config.output_dir.split('/')[-1])
        self.writer = ExprWriter(config, self.generate_expr_name())
        self.optimize_images = OrderedDict()
        self.verbose_keys = []
    
    def prepare_model(self) -> typing.Union[VGG16FeatureExtractor, VGG19FeatureExtractor, InceptionFeatureExtractor]:
        if self.config.model_type == 'vgg19':
            model = VGG19FeatureExtractor(str2list(self.config.layers), self.config.standardize)
        elif self.config.model_type == 'vgg16':
            model = VGG16FeatureExtractor(str2list(self.config.layers), self.config.standardize)
        elif self.config.model_type == 'inception':
            model = InceptionFeatureExtractor()
        model.freeze_params()
        model.to(self.device)
        return model
    
    def prepare_transforms(self):
        """prepare transforms for preprocess image and postprocess image"""
        transform_pre_list = []
        transform_post_list = []
        if not self.config.exact_resize:
            transform_pre_list.append(ResizeMaxSide(self.config.max_side))
        else:
            transform_pre_list.append(transforms.Resize(self.config.exact_size, antialias=True))
        transform_pre_list.append(transforms.ToTensor()) # PIL to Tensor
        if self.config.input_range != 1:
            transform_pre_list.append(transforms.Lambda(lambda x: x.mul_(self.config.input_range)))
            transform_post_list.append(transforms.Lambda(lambda x: x.mul_(1/self.config.input_range)))
        transform_post_list.reverse()
        transform_post_list.append(transforms.Lambda(lambda x: torch.clamp(x, min=0, max=1)))
        # transform_post_list.append(transforms.Lambda(lambda x: (x-x.min()) / (x.max()-x.min())))
        transform_post_list.append(transforms.ToPILImage()) # Tensor to PIL

        return transforms.Compose(transform_pre_list), transforms.Compose(transform_post_list)
    
    def generate_expr_name(self):
        """save basic config to output folder name"""
        infos = []
        infos += [self.config.optimize_strategy]
        # infos += [self.config.model_type]
        infos += [self.config.layers.replace(' ', '').replace(',', '-')]
        # infos += [self.config.optimizer]
        # infos += AorB(self.config.standardize, 'std', 'raw')
        # infos += ['R' + str(self.config.input_range)]
        infos += self.add_extra_infos()

        return '_'.join(infos)
    
    def add_extra_infos(self):
        """save additional infomation to output folder name"""
        return []
    
    def generate_filename(self, content_path: str, style_path:str):
        """save basic infomation to output filename"""
        infos = [get_basename(content_path), get_basename(style_path)]
        infos += self.add_extra_file_infos()
        return '_'.join(infos)
    
    def add_extra_file_infos(self):
        """save additional infomation to output filename"""
        return []
    
    ############################ Tools Start ##################################
    def lerp(self, t_from, t_to, alpha):
        return (1 - alpha) * t_from + alpha * t_to
        
    def tensor2pil(self, t: torch.Tensor):
        return self.transform_post(t.detach().cpu().squeeze())
    
    def save_image(self, t: torch.Tensor, key: str, verbose=False):
        """
        save tensor as image to self.optimize_images dict.\n
        if verbose=False, always save.\n
        if verbose=True, save when self.config.verbose is True.
        """
        if not verbose or self.config.verbose:
            self.optimize_images[key] = self.tensor2pil(t)

    def feat2embedding(self, feats: torch.Tensor):
        return rearrange(feats, 'n c h w -> (h w) (n c)')
    
    def inspect_features(self, feats: torch.Tensor, tag: str):
        LOGGER.debug('%s: mean=%.3e, var=%.3e, max=%.3e, min=%.3e', tag, feats.detach().mean().item(), feats.detach().var().item(), feats.detach().max().item(), feats.detach().min().item())
    
    def visualize_features(self, feats: typing.List[torch.Tensor], tag: str, normalize=True):
        """
        visualize every channel of feature maps with color map "turbo".

        :param feats: List of feature maps [(1,c1,h1,w1), (1,c2,h2,w2), ... ]
        :param tag: image tag name
        """
        for i, fm in enumerate(feats, 1):
            false_color_images = []
            f = fm.detach()
            for ch in f[0]: # ch:[h, w]
                if normalize:
                    # ch = (ch - ch.min())/(ch.max() - ch.min()) # 直接归一化

                    # 以均值为界分别归一化
                    ch = ch - ch.mean()
                    ch[ch < 0] = ch[ch < 0] / abs(ch.min())
                    ch[ch > 0] = ch[ch > 0] / ch.max()
                    ch = (ch + 1)/2
                # applyColorMap转换灰度图需要输入类型为CV_8UC1
                cvin = (ch*255).cpu().numpy().astype(np.uint8) 
                # numpy支持[::-1]，但tensor不支持。并且，使用[::-1]后，并没有真的生成新的numpy数组，而只是，将numpy数组底层的stride记为-1了。这样传给torch.tensor就会报错。==> At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
                cvnp = cv2.applyColorMap(cvin, cv2.COLORMAP_TWILIGHT_SHIFTED)[..., ::-1].copy()  # bgr->rgb
                false_color_images.append(torch.tensor(cvnp).permute(2, 0, 1)) # (h,w,3)->(3,h,w)
            f = torch.stack(false_color_images) # (c,3,h,w)
                
            self.writer.add_images(tag, f, global_step=i, dataformats='NCHW')
            # self.writer.add_images(tag, repeat(rearrange(fm, 'n c h w -> c n h w'), 'n c h w -> n (m c) h w', m=3), global_step=i, dataformats='NCHW')
    
    def get_lap_filtered_image(self, path):
        # 读取图像
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 应用拉普拉斯变换
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        # 由于拉普拉斯变换可能会产生负值，所以将结果转换为绝对值，并转换为8位无符号整型
        laplacian_abs = np.uint8(np.absolute(laplacian))
        if self.config.exact_resize:
            content_laplacian = transforms.Resize(self.config.exact_size, antialias=True)(transforms.ToTensor()(laplacian_abs))
        else:
            content_laplacian = ResizeMaxSide(self.config.max_side)(transforms.ToTensor()(laplacian_abs))
        content_laplacian = repeat(content_laplacian, 'c h w -> (n c) h w', n=3).unsqueeze(0)
        return content_laplacian.to(self.device)
    
    def downsample_features_if_needed(self, feats, split=False, max_size=64):
        # 如果特征图宽高太大，则进行稀疏采样
        if max(feats.size(2), feats.size(3)) > max_size:
            stride = max(feats.size(2), feats.size(3)) // max_size
            if split:
                feats_lst = split_downsample(feats, stride)
                feats = torch.cat(feats_lst, dim=1)
            else:
                offset = random.randint(0, stride - 1)
                feats = feats[:, :, offset::stride, offset::stride]
        return feats
    ############################ Tools End ####################################
       
    def common_optimize_process(self, Ic, Is):
        raise NotImplementedError("you should implement this method in extended class")

    def pyramid_optimize_process(self, Ic, Is):
        raise NotImplementedError("you should implement this method in extended class")

    def cascade_optimize_process(self, Ic, Is):
        raise NotImplementedError("you should implement this method in extended class")

    def __call__(self, content_path: str, style_path: str):
        #! set up folders which will save the output of algorithm and contain some necessary infomation.
        expr_name = self.generate_expr_name()
        filename = self.generate_filename(content_path, style_path)
        expr_dir = self.config.output_dir  + '/' + self.config.name + '/' + expr_name
        check_folder(expr_dir)
        LOGGER.info(f'Output: {expr_dir}')
        LOGGER.info(f'Filename: {filename}')

        self.time_record.reset_time()
        
        #! read inputs and preprocess
        # prepare input tensors: (1, 3, h, w), (1, 3, h, w)
        # convert("RGB") make gray image also 3 channels
        Ic = self.transform_pre(Image.open(content_path).convert("RGB")).unsqueeze(0).to(self.device)
        Is = self.transform_pre(Image.open(style_path).convert("RGB")).unsqueeze(0).to(self.device)
        self.content_path = content_path #* 当前拉普拉斯预处理要用。以后应该删除。
        # Is2 = self.transform_pre(Image.open('data/nnst_style/S4.jpg').convert("RGB")).unsqueeze(0).to(self.device)

        resize_helper = ResizeHelper()
        Ic = resize_helper.resize_to8x(Ic)
        Is = resize_helper.resize_to8x(Is)
        # self.Is2 = resize_helper.resize_to8x(Is2)

        #! Core of Algorithm
        if self.config.optimize_strategy == 'common':
            self.common_optimize_process(Ic, Is)
        elif self.config.optimize_strategy == 'pyramid':
            self.pyramid_optimize_process(Ic, Is)
        else:
            raise NotImplementedError('TODO')
        
        if self.config.is_demo:
            return self.optimize_images['result']

        #! show and save results
        self.time_record.set_record_point('Optimization')

        #! calc metrics
        ssim_metric = ssim(Ic.to(self.device), self.transform_pre(self.optimize_images['result']).to(self.device).unsqueeze(0))
        lpips_metric = self.lpips(Is.to(self.device), self.transform_pre(self.optimize_images['result']).to(self.device).unsqueeze(0))
        # lpips_metric = torch.tensor(-1)
        self.time_record.set_record('SSIM', f'{ssim_metric.item():.4f}')
        self.time_record.set_record('LPIPS', f'{lpips_metric.item():.4f}')

        Ics = self.transform_pre(self.optimize_images['result']).to(self.device).unsqueeze(0)

        mean_loss, std_loss, cov_loss, mean_loss_before, std_loss_before, cov_loss_before = 0, 0, 0, 0, 0, 0

        with torch.no_grad():
            Ics_41, Ics_featrues = self.model(Ics)
            Is_41, Is_featrues = self.model(Is)
            Ic_41, Ic_featrues = self.model(Ic)
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
            # content_loss = torch.mean((Ics_41 - Ic_41)**2)
            
        self.time_record.set_record('Mean Loss', f'{mean_loss:.3e}')
        self.time_record.set_record('L_m Before', f'{mean_loss_before:.3e}')
        self.time_record.set_record('Std Loss', f'{std_loss:.3e}')
        self.time_record.set_record('L_std Before', f'{std_loss_before:.3e}')
        self.time_record.set_record('Cov Loss', f'{cov_loss:.4f}')
        self.time_record.set_record('L_cov Before', f'{cov_loss_before:.4f}')
        # self.time_record.set_record('Content Loss', f'{content_loss:.4f}')
        # self.time_record.set_record('L_c Before', f'{content_loss_before:.4f}')
        self.time_record.show()
        self.time_record.save(expr_dir, filename)

        # torch.save(mean_diffs, expr_dir + f'/{filename}_data.pt')
        
        if self.config.save_process_gif:
            images2gif(list(self.optimize_images.values()), expr_dir + f'/{filename}_process.gif')

        for k, img in self.optimize_images.items():
            img.save(expr_dir + f"/{filename}_{k}.png")

        if self.optimize_images.get('result', None) is not None and self.config.save_rgbhist:
            Io = self.optimize_images['result']
            Io_numpy = np.array(Io)
            fig, ax = plt.subplots()
            bgrColor = ('blue', 'green', 'red')
             # 统计窗口间隔 , 设置小了锯齿状较为明显 最小为1 最好可以被256整除
            bin_win  = 2
            # 设定统计窗口bins的总数
            bin_num = int(256/bin_win)
            # 控制画布的窗口x坐标的稀疏程度. 最密集就设定xticks_win=1
            xticks_win = 8

            for cidx, color in enumerate(bgrColor):
                counts, bins = np.histogram(Io_numpy[:, :, cidx].reshape(-1), bins=bin_num, range=(0, 256))
                ax.plot(counts, color=color)

            # 设定画布的范围
            ax.set_xlim([0, bin_num])
            # 设定x轴方向标注的位置
            ax.set_xticks(np.arange(0, bin_num+1, xticks_win))
            # 设定x轴方向标注的内容
            ax.set_xticklabels(list(range(0, 257, bin_win*xticks_win)),rotation=45)

            # 保存
            plt.savefig(expr_dir + f'/{filename}_{k}_rgbhist.png')

        # change config of wandb to save important infomation
        self.writer.update_cfg(Timecost=self.time_record.get_record_point('Optimization'))


class ExprWriter(object):

    def __init__(self, config: OptimzeBaseConfig, expr_name) -> None:
        self.config = config
        
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        
        if self.config.use_tensorboard:
            self.summary_writer = SummaryWriter(f'log/{self.config.name}/{expr_name}/{timestamp}')

        if self.config.use_wandb:
            wandb.init(project='Optimize Based Style Transfer', # wandb项目名称 
                # group=self.config.name, # 记录的分组，用于过滤
                # job_type=self.config.model_type, # 记录的类型，用于过滤
                name=expr_name + '_' + timestamp, # 自定义run名称
                config=asdict(self.config)) # config必须接受字典对象

    def log(self, d, **kwargs):
        """wandb log"""
        if self.config.use_wandb:
            wandb.log(d, **kwargs)
    
    def log_scaler(self, tag, scalar_value, **kwargs):
        """log single scaler"""
        if self.config.use_wandb:
            wandb.log({tag: scalar_value})

        if self.config.use_tensorboard:
            self.summary_writer.add_scalar(tag, scalar_value, **kwargs)

    def log_image(self, tag, image, **kwargs):
        """log images"""
        if self.config.use_wandb:
            wandb.log({tag: wandb.Image(image, **kwargs)})
    
    def update_cfg(self, **kwargs):
        """update config"""
        if self.config.use_wandb:
            wandb.config.update(kwargs)

    def add_histogram(self, tag, values,**kwargs):
        if self.config.use_tensorboard:
            self.summary_writer.add_histogram(tag, values, **kwargs)

    def add_embedding(self, embeddings, **kwargs):
        if self.config.use_tensorboard:
            self.summary_writer.add_embedding(embeddings, **kwargs)
    
    def add_image(self, tag, image, **kwargs):
        if self.config.use_tensorboard:
            self.summary_writer.add_image(tag, image, **kwargs)
    
    def add_images(self, tag, image, **kwargs):
        if self.config.use_tensorboard:
            self.summary_writer.add_images(tag, image, **kwargs)