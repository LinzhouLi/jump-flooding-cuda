import torch
from . import _C

def jump_flooding(input: torch.Tensor) -> torch.Tensor:
    if input.shape[0] == 1: 
        input = input.squeeze(0)
    if input.shape[-1] == 1: 
        input = input.squeeze(-1)
    return _C.jump_flooding(input)