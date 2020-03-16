# Differentiable Data Augmentation Library

![](https://github.com/moskomule/dda/workflows/pytest/badge.svg)

This library is a core of several internal projects and changing APIs daily.

## Requirements and Installation

### Requirements

```
Python>=3.7
PyTorch>=1.4.0
torchvision>=0.5
kornia>=0.2
hydra>=0.11 # to run examples
```

### Installation

```
pip install -U git+https://github.com/moskomule/dda
```

## APIs

### `dda.functional`

Basic operations that can be differentiable w.r.t. the magnitude parameter `mag`.

```python
def operation(img: torch.Tensor,
              mag: Optional[torch.Tensor]) -> torch.Tensor:
    ...
```

`dda.pil` contains the similar APIs using PIL.

As introduced in Faster AutoAugment, some operations use straight-through estimator for magnitude parameters.

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

Examples directory will contain the following implementations:

### Faster AutoAugment

Coming soon.

### RandAugment (wihtout using differentiable data augmentation)

```shell script
python main.py augment=randaugmnet
```



## Citation

```bibtex
@article{hataya2019,
    title={{Faster AutoAugment: Learning Augmentation Strategies using Backpropagation}},
    author={Ryuichiro Hataya and Jan Zdenek and Kazuki Yoshizoe and Hideki Nakayama},
    year={2019},
    eprint={1911.06987},
    archivePrefix={arXiv},
}
```