import torch


def _shape_check(input: torch.Tensor,
                 v: torch.Tensor) -> None:
    if input.dim() != 4:
        raise RuntimeError(f"`input` is required to be 4D tensor (BxCxHxW), but got {input.size()}")
    if v.dim() == 1:
        v = v.expand(input.size(0), 1)
    # v: Bx2
    if not (input.size(0) == v.size(0) and v.dim() == 2):
        raise RuntimeError(f"`v` is required to be 1D (2) or 2D (Bx2) tensor")


def _affine(input: torch.Tensor,
            matrix: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def _shear(v: torch.Tensor) -> torch.Tensor:
    # generate shearing matrix
    m = torch.eye(3, out=v.new_empty((3, 3)))


def shear(input: torch.Tensor,
          v: torch.Tensor) -> torch.Tensor:
    # input: BxCxHxW
    # v: 2 or Bx2
    _shape_check(input, v)
    matrix = _shear(v)
    return _affine(input, matrix)


def translate(input: torch.Tensor,
              v: torch.Tensor) -> torch.Tensor:
    _shape_check(input, v)


def rotate(input: torch.Tensor,
           v: torch.Tensor) -> torch.Tensor:
    _shape_check(input, v)
