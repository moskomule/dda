import random

from PIL.Image import Image as PILImage

from dda.pil import *


class RandAugment(object):
    def __init__(self,
                 num_aug: int,
                 magnitude: float):
        assert num_aug > 0
        assert 0 < magnitude <= 30
        self.num_aug = num_aug
        self.magnitude = magnitude
        self.operations = self.augment_list()

    def __call__(self,
                 img: PILImage) -> PILImage:
        ops = random.choices(self.operations, k=self.num_aug)
        for op, min_val, max_val in ops:
            v = (self.magnitude / 30) * (max_val - min_val) + min_val
            img = op(img, v)
        return img

    @staticmethod
    def augment_list() -> list:
        return [(auto_contrast, 0, 1),
                (equalize, 0, 1),
                (invert, 0, 1),
                (posterize, 0, 4),
                (solarize, 0, 256),
                (saturate, 0.1, 1.9),
                (contrast, 0.1, 1.9),
                (contrast, 0.1, 1.9),
                (brightness, 0.1, 1.9),
                (sharpness, 0.1, 1.9),
                (rotate, 0, 30),
                (shear_x, 0, 0.3),
                (shear_y, 0, 0.3),
                (translate_x, 0, 0.45),
                (translate_y, 0, 0.45)]
