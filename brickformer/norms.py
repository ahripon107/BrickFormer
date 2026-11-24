from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .registry import NORMS


@NORMS.register_module()
class RMSNorm(nn.Module):
    """RMSNorm computed in float32 (matches HF Llama numerics)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


@NORMS.register_module()
class LayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__(dim, eps=eps, bias=bias)


class RMSNormGated(nn.Module):
    """``RMSNorm(x * silu(gate))`` with optional grouping, as used by Mamba2 and GDN outputs."""

    def __init__(self, dim: int, eps: float = 1e-5, group_size: int | None = None):
        super().__init__()
        self.eps = eps
        self.group_size = group_size or dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        if gate is not None:
            x = x * F.silu(gate.float())
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.group_size)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.view(shape).to(dtype)
