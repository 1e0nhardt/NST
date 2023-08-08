import time
import typing
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import wraps
import cv2
import numpy as np

import torch
from einops import rearrange, repeat
from rich.traceback import install
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import wandb
from models.inception import InceptionFeatureExtractor
from models.vgg import VGG16FeatureExtractor, VGG19FeatureExtractor
from utils import (LOGGER, AorB, ResizeMaxSide, TimeRecorder, check_folder,
                   get_basename, images2gif, str2list)

install(show_locals=False) # 设置rich为默认的异常输出处理程序


@dataclass
class OptimzeBaseConfig:
    """
    基本训练配置
    """

    name: str = 'base'
    """expr name. Change this to your pipeline name."""
    output_dir: str = "./results"
    """output dir"""
    max_iter: int = 500
    """optimize steps"""
    show_iter: int = 50
    """frequencies to show current loss"""
    model_type: str = 'vgg16'
    """feature extractor model type vgg16 | vgg19 | inception. Generally, vgg16 is enough."""
    layers: str = "1,6,11,18,25"
    """layer indices of style features which should seperate by ','"""
    optimizer: str = 'adam'
    """optimizer type: adam | lbfgs"""
    lr: float = 1e-3
    """learning rate"""
    standardize: bool = True
    """use ImageNet mean to standardization"""
    input_range: float = 1
    """scale number range. default 1."""
    exact_resize: bool = False
    """resize use exact size"""
    max_side: int = 512
    """max side length of resized image"""
    exact_size: tuple = (512, 512)
    """exact size of inputs"""
    use_tv_reg: bool = False
    """use total variance regularizer"""
    save_init: bool = False
    """save init image"""
    save_process: bool = False
    """save images of optimize process as gif"""
    verbose: bool = False
    """show additional information"""
    use_wandb: bool = False
    """use wandb"""
    use_tensorboard: bool = False
    """use tensorboard"""


class OptimzeBasePipeline(object):
    def __init__(self, config: OptimzeBaseConfig) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        LOGGER.info(config)

        # prepare model
        self.model = self.prepare_model()
        
        # prepare transforms
        self.transform_pre, self.transform_post = self.prepare_transforms()

        # loggers
        self.time_record = TimeRecorder(self.config.output_dir.split('/')[-1])
        self.writer = ExprWriter(config, self.generate_expr_name())
        self.optimize_images = OrderedDict()
    
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
            transform_pre_list.append(transforms.Resize(self.config.exact_size))
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
        infos += [self.config.model_type]
        infos += [self.config.layers.replace(' ', '').replace(',', '-')]
        infos += [self.config.optimizer]
        infos += AorB(self.config.standardize, 'std', 'raw')
        infos += ['R' + str(self.config.input_range)]
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

    def feat2embedding(self, feats: torch.Tensor):
        return rearrange(feats, 'n c h w -> (h w) (n c)')
    
    def inspect_features(self, feats: torch.Tensor, tag: str):
        LOGGER.debug('%s: mean=%.3e, var=%.3e, max=%.3e, min=%.3e', tag, feats.detach().mean().item(), feats.detach().var().item(), feats.detach().max().item(), feats.detach().min().item())
    
    def visualize_features(self, feats: torch.Tensor, tag: str):
        for i, f in enumerate(feats, 1):
            self.writer.add_images(tag, repeat(rearrange(f, 'n c h w -> c n h w'), 'n c h w -> n (m c) h w', m=3), global_step=i, dataformats='NCHW')
    
    def get_lap_filtered_image(self, path):
        # 读取图像
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 应用拉普拉斯变换
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        # 由于拉普拉斯变换可能会产生负值，所以将结果转换为绝对值，并转换为8位无符号整型
        laplacian_abs = np.uint8(np.absolute(laplacian))
        if self.config.exact_resize:
            content_laplacian = transforms.Resize(self.config.exact_size)(transforms.ToTensor()(laplacian_abs))
        else:
            content_laplacian = ResizeMaxSide(self.config.max_side)(transforms.ToTensor()(laplacian_abs))
        content_laplacian = repeat(content_laplacian, 'c h w -> (n c) h w', n=3).unsqueeze(0)
        return content_laplacian.to(self.device)
    ############################ Tools End ####################################
       
    def optimize_process(self, content_path, style_path):
        raise NotImplementedError("you must implement this method in extended class")

    def __call__(self, content_path: str, style_path: str):
        # set up folders which will save the output of algorithm and contain some necessary infomation.
        expr_name = self.generate_expr_name()
        filename = self.generate_filename(content_path, style_path)
        expr_dir = self.config.output_dir  + '/' + self.config.name + '/' + expr_name
        check_folder(expr_dir)
        LOGGER.info(f'Output: {expr_dir}')
        LOGGER.info(f'Filename: {filename}')

        self.time_record.reset_time()

        #! Core of Algorithm
        self.optimize_process(content_path, style_path)

        # show excution time
        self.time_record.set_record_point('Optimization')
        self.time_record.show()
        
        # save results
        if self.config.save_process:
            images2gif(list(self.optimize_images.values()), expr_dir + f'/{filename}_process.gif')

        if self.config.verbose:
            for k, img in self.optimize_images.items():
                img.save(expr_dir + f'/{filename}_{k}.png')

        if self.config.save_init and self.optimize_images.get('init', None) != None:
            self.optimize_images['init'].save(expr_dir + f'/{filename}_init.png')

        if self.optimize_images.get('result', None) != None:  
            self.optimize_images['result'].save(expr_dir + f'/{filename}_result.png')

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