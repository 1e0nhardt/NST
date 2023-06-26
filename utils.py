from PIL import Image
import numpy as np
import time
import torch
import itertools
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import matplotlib as mpl
import inspect
import os
import torch.nn.functional as F

CONSOLE = Console(width=120)


class PadHelper(object):
    def pad_to8x(self, t:torch.Tensor, mode:str = 'reflect'):
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

    def crop_to_original(self, t:torch.Tensor):
        assert self.t_data is not None, "this method should only use after pad_to8x()"
        h_new, w_new, t_pad, b_pad, l_pad, r_pad = self.t_data
        return t[:, :, t_pad:h_new-b_pad, l_pad:w_new-r_pad]


def showimg(ax, img, title=None, cmap='viridis'):
    if len(img.shape) == 2:
        cmap = 'gray'
    if type(img) is torch.Tensor:
        img = tensor2img(img)
    ax.imshow(img, cmap=cmap)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')


def showimgs(rows, cols, imgs, titles=None, cmap='viridis'):
    dpi = mpl.rcParams['figure.dpi']
    figsize = (400/dpi*cols, 400/dpi*rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    try:
        axes = axes.reshape(-1)
    except AttributeError:
        axes = [axes]
    if titles is None:
        titles = [None] * len(imgs)
    for ax, img, title in zip(axes, imgs, titles):
        showimg(ax, img, title, cmap)
    plt.show()
    

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
    rgb_numpy = tensor2img(rgb_tensor)
    Image.fromarray(rgb_numpy).save(path)


def calc_mem_allocated(t: torch.Tensor):
    assert t.dtype is torch.float32
    n = torch.numel(t)
    mem = n * 4 / 1024 ** 2 # n*4 B => mem MB
    return mem


def print_gpu_mem():
    frameinfo = inspect.stack()[1]
    where_str = frameinfo.filename + ' line ' + str(frameinfo.lineno) + ': ' + frameinfo.function
    print(f'\033[92mAt {where_str:<50}\033[0m \n\033[92mAllocated: {torch.cuda.memory_allocated() / 1024**2} MB, Reserved: {torch.cuda.memory_reserved() / 1024**2} MB\033[0m \n')


def print_module_params(m:torch.nn.Module, only_shape:bool = False):
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
        self._record = {}
        self._start = time.time()
    
    def set_record_point(self, key: str):
        now = time.time()
        if key in self._record.keys():
            CONSOLE.print(f'The key is already in use and will overwrite the previous content ==> {key}')
        self._record[key] = now - self._start
        self._start = now
    
    def show(self):
        table = Table(title=self._title)
        table.add_column('Key', style='cyan')
        table.add_column('Time cost', style='magenta')
        for k, v in self._record.items():
            table.add_row(k, str(v))
        CONSOLE.print(table)


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


if __name__ == '__main__':
    r = Recorder()
    r.add_row('remd', '128x256x256', 64, 3, 3, 72, 1)
