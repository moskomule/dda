import pytest
import torch

from dda import operations


@pytest.fixture
def inputs():
    return [torch.randint(0, 255, (b, 3, 32, 32), dtype=torch.float32) / 255 for b in [1, 2, 4]]


def test_operations(inputs):
    for module in [operations.ShearX(),
                   operations.ShearY(),
                   operations.TranslateY(),
                   operations.TranslateY(),
                   operations.Rotate(),
                   operations.Invert(),
                   operations.HorizontalFlip(),
                   operations.Invert(),
                   operations.Solarize(),
                   operations.Posterize(),
                   operations.Contrast(),
                   operations.Saturate(),
                   operations.Brightness(),
                   operations.Sharpness(),
                   operations.AutoContrast(),
                   operations.Equalize(),
                   operations.SamplePairing()]:
        for input in inputs:
            out = module(input)
            out.mean().backward()
