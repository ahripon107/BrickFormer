from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from ..registry import FFNS


def default_hidden(dim: int, multiple_of: int = 256) -> int:
    hidden = int(8 * dim / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


@FFNS.register_module()
class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))``. ``hidden`` defaults to 8/3 * dim rounded up to ``multiple_of``."""

    def __init__(self, dim: int, hidden: int | None = None, bias: bool = False, multiple_of: int = 256):
        super().__init__()
        hidden = hidden or default_hidden(dim, multiple_of)
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@FFNS.register_module()
class GeLUMLP(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, bias: bool = True, approximate: str = "tanh"):
        super().__init__()
        hidden = hidden or 4 * dim
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)
        self.approximate = approximate

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x), approximate=self.approximate))
