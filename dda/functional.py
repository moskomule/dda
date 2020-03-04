""" `functional` contains deterministic functions
img image tensor `img` is expected to be CxHxW or BxCxHxW and its range should be [0, 1]
`mag=0` expects no transformation

"""

from typing import Optional

import kornia
import torch
from torch.nn import functional as F

from .kernels import get_sharpness_kernel
from .ste import _ste

__all__ = ['shear_x', 'shear_y', 'translate_x', 'translate_y', 'hflip', 'vflip', 'rotate', 'invert', 'solarize',
           'posterize', 'gray', 'contrast', 'auto_contrast', 'saturate', 'brightness', 'hue', 'sample_pairing',
           'equalize', 'sharpness']


# helper functions
# helper functions execpt `_shape_check` assumes img is 4D
def _shape_check(img: torch.Tensor,
                 mag: Optional[torch.Tensor] = None) \
    -> torch.Tensor or (torch.Tensor, torch.Tensor):
    if img.dim() == 3:
        img.unsqueeze(0)
    if mag is None:
        return img
    if mag.nelement() != 1 and mag.size(0) != img.size(0):
        raise RuntimeError('Shape of `mag` is expected to be `1` or `B`')
    return img, mag


def _blend_image(img1: torch.Tensor,
                 img2: torch.Tensor,
                 alpha: torch.Tensor):
    alpha = alpha.view(-1, 1, 1, 1)
    return ((1 - alpha) * img1 + alpha * img2).clamp_(0, 1)


def _gray(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.chunk(3, dim=1)
    return 0.299 * r + 0.587 * g + 0.110 * b


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    return kornia.rgb_to_hsv(img)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    return kornia.hsv_to_rgb(img)


def _blur(img: torch.Tensor,
          kernel: torch.Tensor) -> torch.Tensor:
    assert kernel.ndim == 2
    c = img.size(1)
    return F.conv2d(F.pad(img, (1, 1, 1, 1), 'reflect'),
                    kernel.repeat(c, 1, 1, 1),
                    padding=0,
                    stride=1,
                    groups=c)


# Geometric transformation functions

def shear_x(img: torch.Tensor,
            mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    mag = torch.stack([mag, torch.zeros_like(mag)], dim=1)
    return kornia.shear(img, mag)


def shear_y(img: torch.Tensor,
            mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    mag = torch.stack([torch.zeros_like(mag), mag], dim=1)
    return kornia.shear(img, mag)


def translate_x(img: torch.Tensor,
                mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    mag = torch.stack([mag, torch.zeros_like(mag)], dim=1)
    return kornia.translate(img, mag)


def translate_y(img: torch.Tensor,
                mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    mag = torch.stack([torch.zeros_like(mag), mag], dim=1)
    return kornia.translate(img, mag)


def hflip(img: torch.Tensor,
          _=None) -> torch.Tensor:
    img = _shape_check(img)
    return img.flip(dims=[2])


def vflip(img: torch.Tensor,
          _=None) -> torch.Tensor:
    img = _shape_check(img)
    return img.flip(dims=[3])


def rotate(img: torch.Tensor,
           mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    return kornia.rotate(img, mag)


# Color transformation functions


def invert(img: torch.Tensor,
           _=None) -> torch.Tensor:
    img = _shape_check(img)
    return 1 - img


def solarize(img: torch.Tensor,
             mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    mag = mag.view(-1, 1, 1, 1)
    mask = (img < mag).float()
    return _ste(mask * img + (1 - mask) * (1 - img), mag)


def posterize(img: torch.Tensor,
              mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    mag = mag.view(-1, 1, 1, 1)
    mask = ~(2 ** (1 - mag).mul_(8).floor().long() - 1)
    return _ste((img.long() & mask).float(), mag)


def gray(img: torch.Tensor,
         _=None) -> torch.Tensor:
    img = _shape_check(img)
    return _gray(img).repeat(1, 3, 1, 1)


def contrast(img: torch.Tensor,
             mag: torch.Tensor) -> torch.Tensor:
    mean = _gray(img * 255).flatten(1).mean(dim=1).add_(0.5).floor_().view(-1, 1, 1, 1) / 255
    return _blend_image(img, mean, mag)


def auto_contrast(img: torch.Tensor,
                  _=None) -> torch.Tensor:
    with torch.no_grad():
        # BxCxHxW -> BCxHW
        reshaped = img.flatten(0, 1).flatten(1, 2) * 255
        # BCx1
        min, _ = reshaped.min(dim=1, keepdim=True)
        max, _ = reshaped.max(dim=1, keepdim=True)
        output = (torch.arange(256, device=img.device, dtype=img.dtype) - min) * (255 / (max - min + 0.1))
        output = output.floor_().gather(1, reshaped.long()).reshape_as(img) / 255
    return _ste(output, img)


def saturate(img: torch.Tensor,
             mag: torch.Tensor) -> torch.Tensor:
    # a.k.a. color
    img, mag = _shape_check(img, mag)
    return _blend_image(img, _gray(img), mag)


def brightness(img: torch.Tensor,
               mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    return _blend_image(img, torch.zeros_like(img), mag)


def hue(img: torch.Tensor,
        mag: torch.Tensor) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    h, s, v = _rgb_to_hsv(img).chunk(3, dim=1)
    mag = mag.view(-1, 1, 1, 1)
    _h = (h + mag) % 1
    return _hsv_to_rgb(torch.cat([_h, s, v], dim=1))


def sample_pairing(img: torch.Tensor,
                   mag: torch.Tensor) -> torch.Tensor:
    indices = torch.randperm(img.size(0), device=img.device, dtype=torch.long)
    return _blend_image(img, img[indices], mag)


def equalize(img: torch.Tensor,
             _=None) -> torch.Tensor:
    # see https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py#L319
    with torch.no_grad():
        # BCxHxW
        reshaped = img.clone().flatten(0, 1) * 255
        size = reshaped.size(0)  # BC
        # 0th channel [0-255], 1st channel [256-511], 2nd channel [512-767]...(BC-1)th channel
        shifted = reshaped + 256 * torch.arange(0, size, device=reshaped.device,
                                                dtype=reshaped.dtype).view(-1, 1, 1)
        # channel wise histogram: BCx256
        histogram = shifted.histc(size * 256, 0, size * 256 - 1).view(size, 256)
        # channel wise cdf: BCx256
        cdf = histogram.cumsum(-1)
        # BCx1
        step = ((cdf[:, -1] - histogram[:, -1]) / 255).floor_().view(size, 1)
        # cdf interpolation, BCx256
        cdf = torch.cat([cdf.new_zeros((cdf.size(0), 1)), cdf], dim=1)[:, :256] + (step / 2).floor_()
        # to avoid zero-div, add 0.1
        output = (cdf / (step + 0.1)).floor_().view(-1)[shifted.long()].reshape_as(img) / 255
    return _ste(output, img)


def sharpness(img: torch.Tensor,
              mag: torch.Tensor,
              kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
    img, mag = _shape_check(img, mag)
    if kernel is None:
        kernel = get_sharpness_kernel(img.device)
    return _blend_image(img, _blur(img, kernel), mag)


def cutout(img: torch.Tensor,
           mag: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
