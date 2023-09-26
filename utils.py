from collections import OrderedDict
import inspect
import itertools
import logging
import os
import time
from typing import Sequence

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from sklearn.cluster import AgglomerativeClustering
from torchvision.transforms.functional import (InterpolationMode,
                                               _interpolation_modes_from_int,
                                               get_dimensions, resize)

CONSOLE = Console()


class MyFilter(logging.Filter):
    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self.pys = []  # 记录项目所有的py文件
        for _, _, i in os.walk(os.path.dirname(__file__)):
            self.pys.extend([j for j in i if j.endswith('py')])
        # CONSOLE.print(self.pys)

    def filter(self, record):
        if record.filename in self.pys:
            return True
        return False

FORMAT = "%(message)s"
rich_handler = RichHandler(markup=True)
rich_handler.addFilter(MyFilter())
logging.basicConfig(
    level=logging.DEBUG, format=FORMAT, datefmt="[%X]", handlers=[rich_handler])

LOGGER = logging.getLogger("rich")


def check_folder(path: str):
    if not os.path.exists(path):
        LOGGER.info(f'[red]Directory {path} does not exist, creating...[/]')
        os.makedirs(path, exist_ok=True)


def get_basename(path: str):
    base_name = os.path.basename(path)
    return os.path.splitext(base_name)[0]


def AorB(cond: bool, a, b=None):
    if cond:
        return [str(a)]
    elif b is None:
        return []
    else:
        return [str(b)]


def str2list(s: str):
    return [int(x.strip()) for x in s.split(',')]


###############################################################################
###############################image utils#####################################
###############################################################################

class ResizeMaxSide(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, antialias="True"):
        super().__init__()
        
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if type(self.size) == int or len(self.size) == 1:  # specified size only for the smallest edge
            _, h, w = get_dimensions(img)
            scale_factor = self.size / max(h, w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
        else:  # specified both h and w
            new_h, new_w = self.size
        return resize(img, [new_h, new_w], self.interpolation, None, self.antialias)


def images2gif(images: list, path, duration_ms=500):
    """将PIL图像列表保存为gif"""
    images[0].save(path, save_all=True, append_images=images[1:], optimize=True, duration=duration_ms, loop=0)


class ResizeHelper(object):
    def resize_to8x(self, t: torch.Tensor, mode: str = 'bilinear'):
        assert len(t.shape) == 4, "tensor shape must be [b, c, h, w]"
        h, w = t.shape[2:]
        h_new = (h + 7) // 8 * 8
        w_new = (w + 7) // 8 * 8
        self.old_size = (h, w)
        self.mode = mode
        return F.interpolate(t, (h_new, w_new), mode=mode)
    
    def resize_to_original(self, t: torch.Tensor):
        assert self.old_size is not None, "this method should only use after resize_to8x()"
        return F.interpolate(t, self.old_size, mode=self.mode)

    def pad_to8x(self, t: torch.Tensor, mode: str = 'reflect'):
        assert len(t.shape) == 4, "tensor shape must be [b, c, h, w]"
        h, w = t.shape[2:]
        h_new = (h + 7) // 8 * 8
        w_new = (w + 7) // 8 * 8
        h_pad = h_new - h
        w_pad = w_new - w
        l_pad = w_pad // 2
        r_pad = w_pad - l_pad
        t_pad = h_pad // 2
        b_pad = h_pad - t_pad
        self.t_data = (h_new, w_new, t_pad, b_pad, l_pad, r_pad)
        return F.pad(t, [l_pad, r_pad, t_pad, b_pad], mode=mode)

    def crop_to_original(self, t: torch.Tensor):
        assert self.t_data is not None, "this method should only use after pad_to8x()"
        h_new, w_new, t_pad, b_pad, l_pad, r_pad = self.t_data
        return t[:, :, t_pad:h_new-b_pad, l_pad:w_new-r_pad]


def showimg(ax, img, title=None, cmap='turbo', opencv=False):
    # if len(img.shape) == 2:
    #     cmap = 'gray'
    if type(img) is torch.Tensor:
        img = tensor2img(img, opencv=opencv)
    elif opencv:
        img = img[..., ::-1]
    ax.imshow(img, cmap=cmap)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')


def showimgs(rows, cols, imgs, titles=None, cmap='turbo', opencv=False):
    dpi = mpl.rcParams['figure.dpi']
    figsize = (200/dpi*cols, 200/dpi*rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    try:
        axes = axes.reshape(-1)
    except AttributeError:
        axes = [axes]
    if titles is None:
        titles = [None] * len(imgs)
    for ax, img, title in zip(axes, imgs, titles):
        showimg(ax, img, title, cmap, opencv=opencv)
    plt.show()


def image_grid(images):
  """Return a 1x5 grid of the images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,3))
  titles = ["HM_init", "HM_3", "HM_2", "HM_1", "HM_0"]
  for i in range(5):
    LOGGER.warning(images[i].max())
    LOGGER.warning(images[i].min())
    # Start next subplot.
    plt.subplot(1, 5, i + 1, title=titles[i])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i].squeeze().permute(1, 2, 0))

  return figure


def tensor2img(rgb_tensor: torch.Tensor, opencv=False):
    """
    0-1 ==> 0-255
    rgb_tensor: (1, 3, h, w)
    """
    if opencv:
        return np.clip(rgb_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        return np.clip(rgb_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)


def save_image(rgb_tensor: torch.Tensor, path: str):
    DIR, _ = os.path.split(path)
    if not os.path.exists(DIR) and DIR != '':
        os.makedirs(DIR, exist_ok=True)
    if rgb_tensor.dtype == torch.uint8:
        rgb_numpy = rgb_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    else:
        rgb_numpy = tensor2img(rgb_tensor)
    Image.fromarray(rgb_numpy).save(path)

###############################################################################
###############################other utils#####################################
###############################################################################

def print_node(n, title=None):
    title = "TensorNode" if not title else title
    table = Table(title=title)
    table.add_column('attr')
    table.add_column('value')
    table.add_row("data", str(n.data))
    table.add_row("grad", str(n.grad))
    table.add_row("grad_fn", str(n.grad_fn))
    table.add_row("is_leaf", str(n.is_leaf))
    table.add_row("requires_grad", str(n.requires_grad))
    CONSOLE.print(table)


def calc_mem_allocated(t: torch.Tensor):
    assert t.dtype is torch.float32
    n = torch.numel(t)
    mem = n * 4 / 1024 ** 2 # n*4 B => mem MB
    return mem


def print_gpu_mem():
    frameinfo = inspect.stack()[1]
    where_str = frameinfo.filename + ' line ' + str(frameinfo.lineno) + ': ' + frameinfo.function
    print(f'\033[92mAt {where_str:<50}\033[0m \n\033[92mAllocated: {torch.cuda.memory_allocated() / 1024**2} MB, Reserved: {torch.cuda.memory_reserved() / 1024**2} MB\033[0m \n')


def print_module_params(m:torch.nn.Module, only_shape:bool = True):
    for name, param in m.named_parameters():
        if only_shape:
            CONSOLE.print(f"{name}: {param.shape}")
        else:
            CONSOLE.print(f"{name}: {param}")


def prepare_style_image(image_filename, resize_hw=None, scale_factor=1.0, alpha_color=None):
    pil_image = Image.open(image_filename)
    width, height = pil_image.size

    if resize_hw is not None:
        h, w = resize_hw
        content_long_side = max([h, w])
        if height > width:
            newsize = (int(content_long_side / height * width), content_long_side)
        else:
            newsize = (content_long_side, int(content_long_side / width * height))
        pil_image = pil_image.resize(newsize, resample=Image.LANCZOS)
    
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.LANCZOS)

    image = np.array(pil_image, dtype="uint8")

    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)

    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."

    image = torch.from_numpy(image.astype("float32") / 255.0)

    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
        
    return image


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


class TimeRecorder(object):

    def __init__(self, title="Time Record") -> None:
        self._title = title
        self._record = OrderedDict()
        self._start = time.time()
    
    def reset_time(self):
        self._start = time.time()
        self._record.clear()
    
    def set_record_point(self, key: str):
        now = time.time()
        if key in self._record.keys():
            CONSOLE.print(f'The key is already in use and will overwrite the previous content ==> {key}')
        self._record[key] = f'{now - self._start:.2f}'
        self._start = now
    
    def set_record(self, key, val):
        self._record[key] = val
    
    def get_record(self, key):
        if key in self._record.keys():
            return self._record[key]
        else:
            return 'Null'
    
    def get_record_point(self, key: str):
        if key not in self._record.keys():
            CONSOLE.print(f'The key is not recorded ==> {key}')
            return -1
        return self._record[key]
    
    def show(self):
        table = Table(title=self._title)
        table.add_column('Key', style='cyan')
        table.add_column('Value', style='magenta')
        for k, v in self._record.items():
            table.add_row(k, str(v))
        CONSOLE.print(table)
    
    def save(self, dir, filename):
        if os.path.exists(os.path.join(dir, 'metrics.csv')):
            with open(os.path.join(dir, 'metrics.csv') , 'a') as f:
                vals = list(self._record.values())
                vals.insert(0, filename)
                f.write(','.join(vals) + '\n')
        else:
            with open(os.path.join(dir, 'metrics.csv') , 'a') as f:
                keys = list(self._record.keys())
                keys.insert(0, 'Metrics')
                vals = list(self._record.values())
                vals.insert(0, filename)
                f.write(','.join(keys) + '\n')
                f.write(','.join(vals) + '\n')


class Recorder(object):
    def __init__(self, filename="Record.csv") -> None:
        self._record = {}
        self._file = open(filename, 'a')
    
    def save_record(self, key, val):
        self._record[key] = val
    
    def get_record(self, key):
        if key in self._record.keys():
            return self._record[key]
        else:
            return 'Null'
    
    def add_row(self, *args):
        def preprocess_fn(x):
            if isinstance(x, float):
                x = round(x, 3)
            return str(x)

        lst = [preprocess_fn(x) for x in args]
        self._file.write(','.join(lst) + '\n')
        self._file.flush()
    
    def __del__(self):
        self._file.close()

recorder = Recorder()


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def hierarchy_cluster(data, n_clusters=3, metric='cosine', linkage='complete'):
    # 定义模型并指定要形成的聚类数，例如3
    cluster = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
    return cluster.fit_predict(data)

def cluster_features(features, n_cluster=3):
    b, c, h, w = features.shape
    data = features.reshape(c, -1).detach().cpu().numpy()
    data_label = hierarchy_cluster(data)
    return torch.from_numpy(data_label).to(features.device)

if __name__ == '__main__':
    # r = Recorder()
    # r.add_row('remd', '128x256x256', 64, 3, 3, 72, 1)

    im = generate_perlin_noise_2d((512, 512), (4,4))
    plt.imshow(im, cmap='gray')
    plt.show()
