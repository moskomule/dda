from typing import Tuple

import torch
from torch.autograd import Function


class _STE(Function):
    """ StraightThrough Estimator

    """

    @staticmethod
    def forward(ctx,
                input_forward: torch.Tensor,
                input_backward: torch.Tensor) -> torch.Tensor:
        ctx.shape = input_backward.shape
        return input_forward

    @staticmethod
    def backward(ctx,
                 grad_in: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return None, grad_in.sum_to_size(ctx.shape)


def _ste(input_forward: torch.Tensor,
         input_backward: torch.Tensor) -> torch.Tensor:
    return _STE.apply(input_forward, input_backward)
