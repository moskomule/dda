# pil counter parts

import functools
import random

from PIL import Image, ImageOps, ImageEnhance
from PIL.Image import Image as PILImage
from torchvision.transforms.functional import adjust_hue

__all__ = ['shear_x', 'shear_y', 'translate_x', 'translate_y', 'hflip', 'vflip', 'rotate', 'invert', 'solarize',
           'posterize', 'gray', 'contrast', 'auto_contrast', 'saturate', 'brightness', 'hue',
           'equalize', 'sharpness']


def pil_function(func):
    @functools.wraps(func)
    def inner(img, v):
        if not isinstance(img, PILImage):
            raise RuntimeError(f'img is expected to be PIL.Image, but got {type(img)} instead!')
        return func(img, v)

    return inner


def _random_flip(v: float) -> float:
    return v if random.random() > 0.5 else -v


@pil_function
def shear_x(img: PILImage,
            v: float) -> PILImage:
    v = _random_flip(v)
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


@pil_function
def shear_y(img: PILImage,
            v: float) -> PILImage:
    v = _random_flip(v)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


@pil_function
def translate_x(img: PILImage,
                v: float) -> PILImage:
    v = _random_flip(v)
    v *= img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


@pil_function
def translate_y(img: PILImage,
                v: float) -> PILImage:
    v = _random_flip(v)
    v *= img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


@pil_function
def rotate(img: PILImage,
           v: float) -> PILImage:
    v = _random_flip(v)
    return img.rotate(v)


@pil_function
def hflip(img: PILImage,
          _=None) -> PILImage:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


@pil_function
def vflip(img: PILImage,
          _=None) -> PILImage:
    return img.transpose(Image.FLIP_TOP_BOTTOM)


@pil_function
def invert(img: PILImage,
           _=None) -> PILImage:
    return ImageOps.invert(img)


@pil_function
def solarize(img: PILImage,
             v: float) -> PILImage:
    return ImageOps.solarize(img, v)


@pil_function
def posterize(img: PILImage,
              v: float) -> PILImage:
    return ImageOps.posterize(img, v)


@pil_function
def gray(img: PILImage,
         _=None) -> PILImage:
    return img.convert('L')


@pil_function
def contrast(img: PILImage,
             v: float) -> PILImage:
    return ImageEnhance.Contrast(img).enhance(v)


@pil_function
def auto_contrast(img: PILImage,
                  _=None) -> PILImage:
    return ImageOps.autocontrast(img)


@pil_function
def saturate(img: PILImage,
             v: float) -> PILImage:
    return ImageEnhance.Color(img).enhance(v)


@pil_function
def brightness(img: PILImage,
               v: float) -> PILImage:
    return ImageEnhance.Brightness(img).enhance(v)


@pil_function
def hue(img: PILImage,
        v: float) -> PILImage:
    return adjust_hue(img, v)


@pil_function
def equalize(img: PILImage,
             _=None) -> PILImage:
    return ImageOps.equalize(img)


@pil_function
def sharpness(img: PILImage,
              v: float) -> PILImage:
    return ImageEnhance.Sharpness(img).enhance(v)
