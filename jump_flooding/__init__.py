import torch
from . import _C

def jump_flooding(input) -> torch.Tensor:
    return _C.jump_flooding(input)