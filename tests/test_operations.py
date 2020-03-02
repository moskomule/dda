import pytest
import torch

from dda import operations


@pytest.fixture
def inputs():
    return [torch.randint(0, 255, (b, 3, 32, 32), dtype=torch.float32) / 255 for b in [1, 2, 4]]


def test_operations(inputs):
    for m in [operations.ShearX]:
        module = m()
        for input in inputs:
            out = module(input)
            out.mean().backward()
