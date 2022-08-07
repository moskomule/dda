# Differentiable Data Augmentation Library

![](https://github.com/moskomule/dda/workflows/pytest/badge.svg)

This library is a core of Faster AutoAugment and its descendants. This library is research oriented, and its AIP may change in the near future.

## Requirements and Installation

### Requirements

```
Python>=3.8
PyTorch>=1.5.0
torchvision>=0.6
kornia>=0.2
```

### Installation

```
pip install -U git+https://github.com/moskomule/dda
```

## APIs

### `dda.functional`

Basic operations that can be differentiable w.r.t. the magnitude parameter `mag`. When `mag=0`, no augmentation is applied, and when `mag=1` (and `mag=-1` if it exists), the severest augmentation is applied. As introduced in Faster AutoAugment, some operations use straight-through estimator to be differentiated w.r.t. their magnitude parameters.

```python
def operation(img: torch.Tensor,
              mag: Optional[torch.Tensor]) -> torch.Tensor:
    ...
```

`dda.pil` contains the similar APIs using PIL (not differentiable).


### `dda.operations`

```python
class Operation(nn.Module):
   
    def __init__(self,
                 initial_magnitude: Optional[float] = None,
                 initial_probability: float = 0.5,
                 magnitude_range: Optional[Tuple[float, float]] = None,
                 probability_range: Optional[Tuple[float, float]] = None,
                 temperature: float = 0.1,
                 flip_magnitude: bool = False,
                 magnitude_scale: float = 1,
                 debug: bool = False):
        ...
```

If `magnitude_range=None`, `probability_range=None`, then `magnitude`, `probability` is not Parameter but Buffer, respectively.

`magnitude` moves in `magnitude_scale * magnitude_range`. 
For example, `dda.operations.Rotation` has `magnitude_range=[0, 1]` and `magnitude_scale=30` so that magnitude is between 0 to 30 degrees. 

To differentiate w.r.t. the probability parameter, `RelaxedBernoulli` is used.

## Examples

* [Faster AutoAugment](./faster_autoaugment)
* [RandAugment](./examples)
* [MADAO](./madao)

## Citation

`dda` (except RandAugment) is developed as a core library of the following research projects. 

If you use `dda` in your academic research, please cite `hataya2020a`.

```bibtex
@inproceesings{hataya2020a,
    title={{Faster AutoAugment: Learning Augmentation Strategies using Backpropagation}},
    author={Ryuichiro Hataya and Jan Zdenek and Kazuki Yoshizoe and Hideki Nakayama},
    year={2020},
    booktitle={ECCV}
}

```