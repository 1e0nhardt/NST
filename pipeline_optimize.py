from functools import wraps
import time
import typing
from dataclasses import dataclass, asdict

import torch
from einops import rearrange, repeat
from rich.traceback import install
from torchvision import transforms

from models.inception import InceptionFeatureExtractor
from models.vgg import VGG16FeatureExtractor, VGG19FeatureExtractor
from utils import (AorB, ResizeMaxSide, TimeRecorder, check_folder,
                   get_basename, images2gif, LOGGER, str2list)
from torch.utils.tensorboard import SummaryWriter
import wandb

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
    model_type: str = 'vgg19'
    """feature extractor model type vgg16 | vgg19 | inception"""
    layers: str = "1,6,11,20,29"
    """layer indices of style features which should seperate by ','"""
    optimizer: str = 'adam'
    """optimizer type: adam | lbfgs"""
    lr: float = 1e-3
    """learning rate"""
    use_softmax_smoothing: bool = False
    """use softmax smoothing to avoid peaky activations of low entropy. From <Rethinking and Improving the Robustness of Image Style Transfer>"""
    standardize: bool = True
    """use ImageNet mean to standardization"""
    input_range: float = 1
    """scale number range after normalization and standardization"""
    use_in: bool = False
    """apply instance normalization on all style feature maps"""
    max_side: int = 512
    """max side length of resized image"""
    save_process: bool = False
    """save images of optimize process as gif"""
    verbose: bool = False
    """show additional information"""
    use_wandb: bool = True
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

        self.time_record = TimeRecorder(self.config.output_dir.split('/')[-1])

        self.writer = ExprWriter(config, self.generate_expr_name())
    
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
        transform_pre_list = []
        transform_post_list = []
        if self.config.name == 'nnst':
            transform_pre_list.append(ResizeMaxSide(self.config.max_side))
        else:
            transform_pre_list.append(transforms.Resize((self.config.max_side, self.config.max_side)))
        transform_pre_list.append(transforms.ToTensor()) # PIL to Tensor
        # if self.config.standardize:
        #     transform_pre_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            # transform_post_list.append(transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]))
        if self.config.input_range != 1:
            transform_pre_list.append(transforms.Lambda(lambda x: x.mul_(self.config.input_range)))
            transform_post_list.append(transforms.Lambda(lambda x: x.mul_(1/self.config.input_range)))
        transform_post_list.reverse()
        transform_post_list.append(transforms.Lambda(lambda x: torch.clamp(x, min=0, max=1)))
        # transform_post_list.append(transforms.Lambda(lambda x: (x-x.min()) / (x.max()-x.min())))
        transform_post_list.append(transforms.ToPILImage()) # Tensor to PIL

        return transforms.Compose(transform_pre_list), transforms.Compose(transform_post_list)
    
    def generate_expr_name(self, mask=False):
        infos = []
        infos += [self.config.model_type]
        infos += [self.config.optimizer]
        infos += AorB(self.config.use_softmax_smoothing, 'softmax2D')
        infos += AorB(self.config.standardize, 'std', 'raw')
        infos += ['R' + str(self.config.input_range)]
        infos += AorB(self.config.use_in, 'IN')
        infos += self.add_extra_infos()
        infos += AorB(mask, 'mask')

        return '_'.join(infos)
    
    def add_extra_infos(self):
        return []
    
    def generate_filename(self, content_path: str, style_path:str):
        infos = [get_basename(content_path), get_basename(style_path)]
        infos += AorB(self.config.optimizer != 'lbfgs', str(self.config.lr))
        infos += self.add_extra_file_infos()
        return '_'.join(infos)
    
    def add_extra_file_infos(self):
        return []
    
    def feat2embedding(self, feats):
        return rearrange(feats, 'n c h w -> (h w) (n c)')
    
    def inspect_features(self, feats, name):
        LOGGER.debug('%s: mean=%.3e, var=%.3e, max=%.3e, min=%.3e', name, feats.detach().mean().item(), feats.detach().var().item(), feats.detach().max().item(), feats.detach().min().item())
       
    def optimize_process(self, content_path, style_path, mask=False):
        raise NotImplementedError("you must implement this method in extended class")

    def __call__(self, content_path: str, style_path: str, mask=False):
        expr_name = self.generate_expr_name(mask=mask)
        filename = self.generate_filename(content_path, style_path)
        expr_dir = self.config.output_dir + '/' + expr_name
        check_folder(expr_dir)
        LOGGER.info(f'Output: {expr_dir}')
        LOGGER.info(f'Filename: {filename}')

        self.time_record.reset_time()

        optimize_images = self.optimize_process(content_path, style_path, mask)

        self.time_record.set_record_point('Optimization')
        self.time_record.show()
        
        if self.config.save_process:
            images2gif(list(optimize_images.values()), expr_dir + f'/{filename}_process.gif')

        if hasattr(self.config, 'max_scales') or self.config.verbose: # NNST
            for k, img in optimize_images.items():
                img.save(expr_dir + f'/{filename}_{k}.png')

        if optimize_images.get('init', None) != None:
            optimize_images['init'].save(expr_dir + f'/{filename}_init.png')
        optimize_images['result'].save(expr_dir + f'/{filename}_result.png')

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