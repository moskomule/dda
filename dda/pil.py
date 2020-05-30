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
    # check if input is PILImage or not
    @functools.wraps(func)
    def inner(*args):
        if not isinstance(args[0], PILImage):
            raise RuntimeError(f'img is expected to be PIL.Image, but got {type(args[0])} instead!')
        return func(*args)

    return inner


def _random_flip(v: float,
                 flip: bool) -> float:
    if not flip:
        return v
    return v if random.random() > 0.5 else -v


@pil_function
def shear_x(img: PILImage,
            v: float,
            flip=True) -> PILImage:
    v = _random_flip(v, flip)
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


@pil_function
def shear_y(img: PILImage,
            v: float,
            flip=True) -> PILImage:
    v = _random_flip(v, flip)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


@pil_function
def translate_x(img: PILImage,
                v: float,
                flip=True) -> PILImage:
    v = _random_flip(v, flip)
    v *= img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


@pil_function
def translate_y(img: PILImage,
                v: float,
                flip=True) -> PILImage:
    v = _random_flip(v, flip)
    v *= img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


@pil_function
def rotate(img: PILImage,
           v: float,
           flip=True) -> PILImage:
    v = _random_flip(v, flip)
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
    return ImageOps.solarize(img, int(255 * v))


@pil_function
def posterize(img: PILImage,
              v: float) -> PILImage:
    return ImageOps.posterize(img, int(v))


@pil_function
def gray(img: PILImage,
         _=None) -> PILImage:
    return img.convert('L')


@pil_function
def contrast(img: PILImage,
             v: float,
             flip=True) -> PILImage:
    # v: 0 to 1
    v = _random_flip(v, flip)
    return ImageEnhance.Contrast(img).enhance(1 - v)


@pil_function
def auto_contrast(img: PILImage,
                  _=None) -> PILImage:
    return ImageOps.autocontrast(img)


@pil_function
def saturate(img: PILImage,
             v: float,
             flip=True) -> PILImage:
    # v: 0 to 1
    v = _random_flip(v, flip)
    return ImageEnhance.Color(img).enhance(1 - v)


@pil_function
def brightness(img: PILImage,
               v: float,
               flip=True) -> PILImage:
    # v: 0 to 1
    v = _random_flip(v, flip)
    return ImageEnhance.Brightness(img).enhance(1 - v)


@pil_function
def hue(img: PILImage,
        v: float) -> PILImage:
    # v: -1 to 1
    return adjust_hue(img, v)


@pil_function
def equalize(img: PILImage,
             _=None) -> PILImage:
    return ImageOps.equalize(img)


@pil_function
def sharpness(img: PILImage,
              v: float,
              flip=True) -> PILImage:
    # v: 0 to 1
    v = _random_flip(v, flip)
    return ImageEnhance.Sharpness(img).enhance(1 - v)
