import pytest
import torch
from PIL import Image

from dda.pil import *


@pytest.fixture
def input():
    return Image.fromarray(torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy())


def test_pils(input):
    for op, min, max in [(auto_contrast, 0, 1),
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
                         (translate_y, 0, 0.45),
                         ]:
        op(input, (min + max) / 2)
