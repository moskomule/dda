""" Kernels for Sharpness or Blur...

"""

from typing import Optional

import torch


def get_sharpness_kernel(device: Optional[torch.device] = None) -> torch.Tensor:
    kernel = torch.ones(3, 3, device=device)
    kernel[1, 1] = 5
    kernel /= 13
    return kernel
