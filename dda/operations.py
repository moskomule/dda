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
           'Hue', 'SamplePairing', 'Equalize']


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
                 operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 initial_magnitude: Optional[float] = None,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = None,
                 probability_range: Optional[Tuple[float, float]] = None,
                 temperature: float = 0.1,
                 flip_magnitude: bool = False,
                 magnitude_scale: float = 1,
                 debug: bool = False):

        super(_Operation, self).__init__()
        self.operation = lambda img, mag: operation(img, mag).clamp_(0, 1)

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

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """

        :param input: torch.Tensor in [0, 1]
        :return: torch.Tensor in [0, 1]
        """

        if self.debug:
            if (input < 0 or input > 1).any():
                raise RuntimeError('Range of `img` is expected to be [0, 1]')

        mask = self.get_mask(input.size(0))
        mag = self.magnitude

        if self.flip_magnitude:
            # (0 or 1) -> (0 or 2) -> (-1 or 1)
            mag = torch.randint(2, (input.size(0),), dtype=torch.float32, device=input.device).mul_(2).sub_(1) * mag

        if self.training:
            return (mask * self.operation(input, mag) + (1 - mask) * input).clamp(0, 1)
        else:
            output = input.clone()
            output[mask == 1] = self.operation(output[mask == 1], mag)
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
            mag.clamp(*self.magnitude_range)
        return mag * self.magnitude_scale

    @property
    def probability(self) -> torch.Tensor:
        if self.probability_range is None:
            return self._probability
        return self._probability.clamp(*self.probability_range)

    def __repr__(self) -> str:
        s = self.__class__.__name__
        prob_state = 'frozen' if self.probability_range is None else 'learnable'
        s += f"( probability={self.probability.item():.3f} ({prob_state}), \n"
        if self.magnitude is not None:
            mag_state = 'frozen' if self.magnitude_range is None else 'learnable'
            s += f"  magnitude={self.magnitude.item():.3f} ({mag_state}),\n"
        s += f" temperature={self.temprature.item():.3f})"
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
                                       probability_range, temperature, debug=debug)


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
                                       probability_range, temperature, debug=debug)


class Brightness(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Brightness, self).__init__(brightness, initial_magnitude, initial_probability, magnitude_range,
                                         probability_range, temperature, debug=debug)


class Hue(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        super(Hue, self).__init__(hue, initial_magnitude, initial_probability, magnitude_range,
                                  probability_range, temperature, debug=debug)


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


class Sharpness(_Operation):
    def __init__(self,
                 initial_magnitude: float = 0.5,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = (0, 1),
                 probability_range: Optional[Tuple[float, float]] = (0, 1),
                 temperature: float = 0.1,
                 debug: bool = False):
        # dummy function
        super(Sharpness, self).__init__(lambda img, mag: sharpness(img, mag, self.kernel), initial_magnitude,
                                        initial_probability, magnitude_range,
                                        probability_range, temperature, debug=debug)
        self.register_buffer('kernel', get_sharpness_kernel())