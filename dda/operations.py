""" Operations

"""

from typing import Optional, Callable, Tuple

import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, Bernoulli

from .functional import (shear_x, shear_y, translate_x, translate_y, hflip, vflip, rotate, invert, solarize, posterize,
                         gray, contrast, auto_contrast, saturate, brightness, hue, sample_pairing, equalize, sharpness)
from .kernels import get_sharpness_kernel

__all__ = ['ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'HorizontalFlip', 'VerticalFlip', 'Rotate',
           'Invert', 'Solarize', 'Posterize', 'Gray', 'Contrast', 'AutoContrast', 'Saturate', 'Brightness',
           'Hue', 'SamplePairing', 'Equalize', 'Sharpness']


class _Operation(nn.Module):
    """ Base class of operation

    :param operation:
    :param initial_magnitude:
    :param initial_probability:
    :param magnitude_range:
    :param probability_range:
    :param temperature: Temperature for RelaxedBernoulli distribution used during training
    :param flip_magnitude: Should be True for geometric
    :param debug: If True, check if img image is in [0, 1]
    """

    def __init__(self,
                 operation: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                 initial_magnitude: Optional[float] = None,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = None,
                 probability_range: Optional[Tuple[float, float]] = None,
                 temperature: float = 0.1,
                 flip_magnitude: bool = False,
                 magnitude_scale: float = 1,
                 debug: bool = False):

        super(_Operation, self).__init__()
        self.operation = operation

        self.magnitude_range = None
        if initial_magnitude is None:
            self._magnitude = None
        elif magnitude_range is None:
            self.register_buffer("_magnitude", torch.empty(1).fill_(initial_magnitude))
        else:
            self._magnitude = nn.Parameter(torch.empty(1).fill_(initial_magnitude))
            assert 0 <= magnitude_range[0] < magnitude_range[1] <= 1
            self.magnitude_range = magnitude_range

        self.probability_range = probability_range
        if self.probability_range is None:
            self.register_buffer("_probability", torch.empty(1).fill_(initial_probability))
        else:
            assert 0 <= initial_probability <= 1
            assert 0 <= self.probability_range[0] < self.probability_range[1] <= 1
            self._probability = nn.Parameter(torch.empty(1).fill_(initial_probability))

        assert 0 < temperature
        self.register_buffer("temperature", torch.empty(1).fill_(temperature))

        self.flip_magnitude = flip_magnitude and (self._magnitude is not None)

        assert 0 < magnitude_scale
        self.magnitude_scale = magnitude_scale
        self.debug = debug

        # to avoid accessing CUDA tensors in multiprocessing env.
        self._py_magnitude = initial_magnitude
        self._py_probability = initial_probability

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """

        :param input: torch.Tensor in [0, 1]
        :return: torch.Tensor in [0, 1]
        """

        mask = self.get_mask(input.size(0))
        mag = self.magnitude

        if self.flip_magnitude:
            # (0 or 1) -> (0 or 2) -> (-1 or 1)
            mag = torch.randint(2, (input.size(0),), dtype=torch.float32, device=input.device).mul_(2).sub_(1) * mag

        if self.training:
            return (mask * self.operation(input, mag) + (1 - mask) * input).clamp_(0, 1)
        else:
            mask.squeeze_()
            output = input
            num_valid = mask.sum().long()
            if torch.is_tensor(mag):
                if mag.size(0) == 1:
                    mag = mag.repeat(num_valid)
                else:
                    mag = mag[mask == 1]
            if num_valid > 0:
                output[mask == 1, ...] = self.operation(output[mask == 1, ...], mag)
            return output.clamp(0, 1)

    def get_mask(self,
                 batch_size=None) -> torch.Tensor:
        size = (batch_size, 1, 1)
        if self.training:
            return RelaxedBernoulli(self.temperature, self.probability).rsample(size)
        else:
            return Bernoulli(self.probability).sample(size)

    @property
    def magnitude(self) -> Optional[torch.Tensor]:
        if self._magnitude is None:
            return None
        mag = self._magnitude
        if self.magnitude_range is not None:
            mag = mag.clamp(*self.magnitude_range)
        m = mag * self.magnitude_scale
        self._py_magnitude = m.item()
        return m

    @property
    def probability(self) -> torch.Tensor:
        if self.probability_range is None:
            return self._probability
        p = self._probability.clamp(*self.probability_range)
        self._py_probability = p.item()
        return p

    def __repr__(self) -> str:
        s = self.__class__.__name__
        prob_state = 'frozen' if self.probability_range is None else 'learnable'
        s += f"(probability={self._py_probability:.3f} ({prob_state}), "
        if self.magnitude is not None:
            mag_state = 'frozen' if self.magnitude_range is None else 'learnable'
            s += f"{' ' * len(s)} magnitude={self._py_magnitude:.3f} ({mag_state}), "
        s += f"{' ' * len(s)} temperature={self.temperature.item():.3f})"
        return s


# Geometric Operations

class ShearX(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 magnitude_scale: float = 0.3,
                 debug: bool = False):
        super(ShearX, self).__init__(shear_x, initial_magnitude, initial_probability, magnitude_range,
                                     probability_range, temperature, flip_magnitude=True,
                                     magnitude_scale=magnitude_scale, debug=debug)


class ShearY(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 magnitude_scale: float = 0.3,
                 debug: bool = False):
        super(ShearY, self).__init__(shear_y, initial_magnitude, initial_probability, magnitude_range,
                                     probability_range, temperature, flip_magnitude=True,
                                     magnitude_scale=magnitude_scale, debug=debug)


class TranslateX(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 magnitude_scale: float = 0.45,
                 debug: bool = False):
        super(TranslateX, self).__init__(translate_x, initial_magnitude, initial_probability, magnitude_range,
                                         probability_range, temperature, flip_magnitude=True,
                                         magnitude_scale=magnitude_scale, debug=debug)


class TranslateY(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 magnitude_scale: float = 0.45,
                 debug: bool = False):
        super(TranslateY, self).__init__(translate_y, initial_magnitude, initial_probability, magnitude_range,
                                         probability_range, temperature, flip_magnitude=True,
                                         magnitude_scale=magnitude_scale, debug=debug)


class HorizontalFlip(_Operation):
    def __init__(self,
                 initial_probability: float = 0.5,
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(HorizontalFlip, self).__init__(hflip, None, initial_probability, None,
                                             probability_range, temperature, debug=debug)


class VerticalFlip(_Operation):
    def __init__(self,
                 initial_probability: float = 0.5,
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(VerticalFlip, self).__init__(vflip, None, initial_probability, None,
                                           probability_range, temperature, debug=debug)


class Rotate(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 magnitude_scale: float = 30,
                 debug: bool = False):
        super(Rotate, self).__init__(rotate, initial_magnitude, initial_probability, magnitude_range,
                                     probability_range, temperature, flip_magnitude=True,
                                     magnitude_scale=magnitude_scale, debug=debug)


# Color Enhancing Operations


class Invert(_Operation):
    def __init__(self,
                 initial_probability: float = 0.5,
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Invert, self).__init__(invert, None, initial_probability, None,
                                     probability_range, temperature, debug=debug)


class Solarize(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Solarize, self).__init__(solarize, initial_magnitude, initial_probability, magnitude_range,
                                       probability_range, temperature, debug=debug)


class Posterize(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Posterize, self).__init__(posterize, initial_magnitude, initial_probability, magnitude_range,
                                        probability_range, temperature, debug=debug)


class Gray(_Operation):
    def __init__(self,
                 initial_probability: float = 0.5,
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Gray, self).__init__(gray, None, initial_probability, None,
                                   probability_range, temperature, debug=debug)


class Contrast(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Contrast, self).__init__(contrast, initial_magnitude, initial_probability, magnitude_range,
                                       probability_range, temperature, flip_magnitude=True, debug=debug)


class AutoContrast(_Operation):
    def __init__(self,
                 initial_probability: float = 0.5,
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(AutoContrast, self).__init__(auto_contrast, None, initial_probability, None,
                                           probability_range, temperature, debug=debug)


class Saturate(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Saturate, self).__init__(saturate, initial_magnitude, initial_probability, magnitude_range,
                                       probability_range, temperature, flip_magnitude=True, debug=debug)


class Brightness(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Brightness, self).__init__(brightness, initial_magnitude, initial_probability, magnitude_range,
                                         probability_range, temperature, flip_magnitude=True, debug=debug)


class Hue(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 magnitude_scale: float = 2,
                 debug: bool = False):
        super(Hue, self).__init__(hue, initial_magnitude, initial_probability, magnitude_range,
                                  probability_range, temperature, magnitude_scale=magnitude_scale, debug=debug)


class SamplePairing(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(SamplePairing, self).__init__(sample_pairing, initial_magnitude, initial_probability, magnitude_range,
                                            probability_range, temperature, debug=debug)


class Equalize(_Operation):
    def __init__(self,
                 initial_probability: float = 0.5,
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Equalize, self).__init__(equalize, None, initial_probability, None,
                                       probability_range, temperature, debug=debug)


class _KernelOperation(_Operation):
    def __init__(self,
                 operation: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                 kernel: torch.Tensor,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 flip_magnitude: bool = False,
                 magnitude_scale: float = 1,
                 debug: bool = False
                 ):
        super(_KernelOperation, self).__init__(None, initial_magnitude,
                                               initial_probability, magnitude_range,
                                               probability_range, temperature, flip_magnitude=flip_magnitude,
                                               magnitude_scale=magnitude_scale, debug=debug)

        # to use kernel properly, this is an ugly way...
        self.register_buffer('kernel', kernel)
        self._original_operation = operation
        self.operation = self._operation

    def _operation(self,
                   img: torch.Tensor,
                   mag: torch.Tensor) -> torch.Tensor:
        return self._original_operation(img, mag, self.kernel)


class Sharpness(_KernelOperation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Sharpness, self).__init__(sharpness, get_sharpness_kernel(), initial_magnitude, initial_probability,
                                        magnitude_range, probability_range, temperature,
                                        flip_magnitude=True, debug=debug)
