from __future__ import annotations

import contextlib
import random
from copy import deepcopy
from typing import Optional

import torch
from PIL.Image import Image as PILImage
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from dda.operations import *
from dda.operations import _Operation, _KernelOperation
from dda import functional, pil

FUNCTIONAL_TO_PIL = {functional.auto_contrast: (pil.auto_contrast, 0, 1),
                     functional.equalize: (pil.equalize, 0, 1),
                     functional.invert: (pil.invert, 0, 1),
                     functional.posterize: (pil.posterize, 0, 1),
                     functional.solarize: (pil.solarize, 0, 256),
                     functional.saturate: (pil.saturate, 0, 2),
                     functional.contrast: (pil.contrast, 0, 2),
                     functional.brightness: (pil.brightness, 0, 2),
                     functional.rotate: (pil.rotate, 0, 30),
                     functional.hflip: (pil.hflip, 0, 1),
                     functional.shear_x: (pil.shear_x, 0, 0.3),
                     functional.shear_y: (pil.shear_y, 0, 0.3),
                     functional.translate_x: (pil.translate_x, 0, 0.45),
                     functional.translate_y: (pil.translate_y, 0, 0.45),
                     functional.sharpness: (pil.sharpness, 0.01, 1.99)}


def pil_forward(self: _Operation,
                img: PILImage):
    if isinstance(self, _KernelOperation):
        _pil_func, _, _ = FUNCTIONAL_TO_PIL[self._original_operation.operation]
    else:
        _pil_func, _, _ = FUNCTIONAL_TO_PIL[self.operation]
    mag = self.py_magnitude
    if self.flip_magnitude:
        if random.random() > 0.5:
            mag = - mag
    prob = self.py_probability
    if random.random() <= prob:
        return _pil_func(img, mag)
    else:
        return img


class SubPolicyStage(nn.Module):
    def __init__(self,
                 operations: nn.ModuleList,
                 temperature: float,
                 soft_select: bool = False
                 ):
        super(SubPolicyStage, self).__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self._weights_list = self._weights.tolist()  # to avoid CUDA initialization error
        self.temperature = temperature
        self.soft_select = soft_select

    def update_py_values(self):
        for op in self.operations:
            op.update_py_values()
        self.weights

    def forward(self,
                input: Tensor
                ) -> Tensor:
        if self.training:
            if self.soft_select:
                return (torch.stack([op(input) for op in self.operations]) * self.weights.view(-1, 1, 1, 1, 1)).sum(0)
            else:
                onehot = F.gumbel_softmax(self.weights, tau=self.temperature, hard=True)
                idx = onehot.argmax().item()
                return self.operations[idx](input) * onehot[idx]
        else:
            return self.operations[Categorical(self.weights).sample()](input)

    @property
    def weights(self
                ):
        w = self._weights.div(self.temperature).softmax(0)
        self._weights_list = w.tolist()
        return w

    def pil_forward(self,
                    img: PILImage):
        i, = random.choices(range(len(self.operations)), self._weights_list)
        return pil_forward(self.operations[i], img)

    def __repr__(self) -> str:
        self.update_py_values()
        weights = " ".join([f"{i:.2f}" for i in self._weights_list])
        ops = [ops for ops in self.operations]
        return f"{weights}\n {ops}"


class SubPolicy(nn.Module):
    def __init__(self,
                 sub_policy_stage: SubPolicyStage,
                 operation_count: int,
                 ):
        super(SubPolicy, self).__init__()
        self.stages = nn.ModuleList([deepcopy(sub_policy_stage) for _ in range(operation_count)])

    def forward(self,
                input: Tensor
                ) -> Tensor:
        for stage in self.stages:
            input = stage(input)
        return input

    def pil_forward(self,
                    img: PILImage):
        for stage in self.stages:
            img = stage.pil_forward(img)
        return img

    def update_py_values(self):
        for stage in self.stages:
            stage.update_py_values()

    def __repr__(self):
        return str(self.stages)


class Policy(nn.Module):
    def __init__(self,
                 operations: nn.ModuleList,
                 num_sub_policies: int,
                 temperature: float = 1,
                 operation_count: int = 2,
                 num_chunks: int = 4,
                 mean: Optional[Tensor] = None,
                 std: Optional[Tensor] = None,
                 random_init: bool = False,
                 soft_select: bool = False
                 ):
        super(Policy, self).__init__()
        self.sub_policies = nn.ModuleList([
            SubPolicy(SubPolicyStage(operations, temperature, soft_select=soft_select), operation_count)
            for _ in range(num_sub_policies)
        ])
        self.num_sub_policies = num_sub_policies
        self.temperature = temperature
        self.operation_count = operation_count
        self.num_chunks = num_chunks
        # if mean is None:
        #     mean, std = torch.ones(3) * 0.5, torch.ones(3) * 0.5
        self.register_buffer('_mean', mean)
        self.register_buffer('_std', std)
        self._pil_mode = False

        if random_init:
            for n, p in self.named_parameters():
                if "prob" in n or "mag" in n:
                    nn.init.uniform_(p, 0.1, 0.9)

    @contextlib.contextmanager
    def pil_mode(self):
        self.update_py_values()
        self._pil_mode = True
        yield
        self._pil_mode = False

    def forward(self,
                input: Tensor,
                ) -> Tensor:

        if self._pil_mode or not self.training:
            # data augmentation is applied by PIL
            # or testing
            return self.normalize(input)

        if self.num_chunks > 1:
            out = [self._forward(inp) for inp in input.chunk(self.num_chunks)]
            x = torch.cat(out, dim=0)
        else:
            x = self._forward(input)

        return self.normalize(x)

    def pil_forward(self, img: PILImage):
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index].pil_forward(img)

    def update_py_values(self):
        for sub in self.sub_policies:
            sub.update_py_values()

    def _forward(self,
                 input: Tensor
                 ) -> Tensor:
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](input)

    def normalize(self,
                  input: Tensor
                  ) -> Tensor:
        # [0, 1] -> [-1, 1]
        return (input - self._mean[:, None, None]) / self._std[:, None, None]

    @staticmethod
    def dda_operations():
        mag = 0.5
        out = [
            ShearX(initial_magnitude=mag),
            ShearX(initial_magnitude=mag),
            ShearY(initial_magnitude=mag),
            TranslateY(initial_magnitude=mag),
            TranslateY(initial_magnitude=mag),
            Rotate(initial_magnitude=mag),
            Invert(),
            Solarize(initial_magnitude=mag),
            Posterize(initial_magnitude=mag),
            Contrast(initial_magnitude=mag),
            Saturate(initial_magnitude=mag),
            Brightness(initial_magnitude=mag),
            Sharpness(initial_magnitude=mag),
            AutoContrast(),
            Equalize(),
        ]
        return out

    @staticmethod
    def madao_policy(num_sub_policies: int = 1,
                     temperature: float = 1,
                     operation_count: int = 2,
                     num_chunks: int = 32,
                     mean: Optional[torch.Tensor] = None,
                     std: Optional[torch.Tensor] = None,
                     ) -> Policy:
        operations = Policy.dda_operations()

        return Policy(nn.ModuleList(operations), num_sub_policies, temperature, operation_count, num_chunks,
                      mean=mean, std=std)

    def __repr__(self):
        s = ""
        for sub_policy in self.sub_policies:
            s += f"{sub_policy}\n"
        s = s[:-1]
        return f"Policy({s})"
