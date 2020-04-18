import pytest
import torch

from dda import functional


@pytest.mark.parametrize('f', [functional.shear_x,
                               functional.shear_y,
                               functional.translate_x,
                               functional.translate_y,
                               functional.rotate,
                               functional.solarize,
                               functional.posterize,
                               functional.contrast,
                               functional.saturate,
                               functional.brightness,
                               functional.hue,
                               functional.sample_pairing,
                               functional.sharpness,
                               functional.gaussian_blur3x3]
                         )
@pytest.mark.parametrize('i', [torch.randint(0, 255, (b, 3, 32, 32), dtype=torch.float32) / 255 for b in [1, 2, 4]])
@pytest.mark.parametrize('v', [torch.rand(s) for s in [1, 2, 4]])
def test_function_with_magnitude(f, i, v):
    i = i.clone()
    i.requires_grad_(True)
    f(i, v).mean().backward()


@pytest.mark.parametrize('f', [functional.vflip,
                               functional.hflip,
                               functional.invert,
                               functional.gray,
                               functional.auto_contrast,
                               functional.equalize])
@pytest.mark.parametrize('i', [torch.randint(0, 255, (b, 3, 32, 32), dtype=torch.float32) / 255 for b in [1, 2, 4]])
def test_function_without_magnitude(f, i):
    f(i, None)
