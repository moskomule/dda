import pytest
import torch

from dda import functional


@pytest.fixture
def inputs():
    return [torch.randint(0, 255, (b, 3, 32, 32), dtype=torch.float32) / 255 for b in [1, 2, 4]]


@pytest.fixture
def vs():
    return [torch.rand(s, requires_grad=True) for s in [1, 2, 4]]


@pytest.fixture
def fs_with_mag():
    return [functional.shear_x,
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
            functional.sharpness]


@pytest.fixture
def fs_without_mag():
    return [functional.vflip,
            functional.hflip,
            functional.invert,
            functional.gray,
            functional.auto_contrast,
            functional.equalize]


def test_function_with_magnitude(fs_with_mag, inputs, vs):
    for f in fs_with_mag:
        for input, v in zip(inputs, vs):
            f(input, v).mean().backward()


def test_function_without_magnitude(fs_without_mag, inputs):
    for f in fs_without_mag:
        for input in inputs:
            f(input)
