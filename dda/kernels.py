""" Kernels for Sharpness or Blur...

"""

from typing import Optional

import torch


def get_sharpness_kernel(device: Optional[torch.device] = None) -> torch.Tensor:
    kernel = torch.ones(3, 3, device=device)
    kernel[1, 1] = 5
    kernel /= 13
    return kernel


def _gaussian(sigma: torch.Tensor,
              kernel_size: int,
              device: Optional[torch.device] = None) -> torch.Tensor:
    radius = kernel_size // 2
    # because functional.tensor_function automatically broadcast
    sigma = sigma.mean().pow(2)
    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32).pow(2).view(-1, 1)
    kernel = kernel + kernel.t()
    kernel = (-kernel / (2 * sigma)).exp()
    return (kernel / kernel.sum()).to(device=device)


def get_gaussian_3x3kernel(sigma: torch.Tensor,
                           device: Optional[torch.device] = None) -> torch.Tensor:
    return _gaussian(sigma, 3, device)


def get_gaussian_5x5kernel(sigma: torch.Tensor,
                           device: Optional[torch.device] = None) -> torch.Tensor:
    return _gaussian(sigma, 5, device)
