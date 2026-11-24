"""The common mixer interface.

Every token mixer implements::

    y, new_state = mixer(x, state=None, mask=None, position_ids=None)
    state = mixer.init_state(batch_size, max_len, device, dtype)

* ``x``: ``[B, T, dim]``.
* ``state``: ``None`` for stateless (training) forward, otherwise the mixer's own
  state object from ``init_state``. It may be updated in place; the returned
  ``new_state`` is what the caller must keep.
* ``mask``: optional key-padding mask ``[B, T]`` (True = real token). Only
  supported without a state.
* ``position_ids``: ``[B, T]`` absolute positions (used by RoPE mixers).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class Mixer(nn.Module):
    dim: int

    def init_state(self, batch_size: int, max_len: int, device=None, dtype=None) -> Any:
        raise NotImplementedError

    def forward(self, x, state=None, mask=None, position_ids=None):  # pragma: no cover
        raise NotImplementedError

    def extra_flops_per_token(self, seq_len: int) -> float:
        """Forward FLOPs per token *not* accounted for by linear layers (e.g. attention scores)."""
        return 0.0


class CausalConv1d(nn.Module):
    """Depthwise causal conv with a rolling state of the last ``kernel_size - 1`` inputs."""

    def __init__(self, channels: int, kernel_size: int = 4, bias: bool = True, activation: bool = True):
        super().__init__()
        self.channels, self.kernel_size, self.activation = channels, kernel_size, activation
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.channels, self.kernel_size - 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """``x``: [B, T, C]. Returns ``(y [B, T, C], new_state [B, C, K-1])``."""
        xt = x.transpose(1, 2)
        if state is None:
            state = xt.new_zeros(x.shape[0], self.channels, self.kernel_size - 1)
        xt = torch.cat([state.to(xt.dtype), xt], dim=-1)
        new_state = xt[..., -(self.kernel_size - 1):]
        y = F.conv1d(xt, self.weight.to(xt.dtype), None if self.bias is None else self.bias.to(xt.dtype),
                     groups=self.channels)
        if self.activation:
            y = F.silu(y)
        return y.transpose(1, 2), new_state


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def init_dt_bias(n: int, dt_min: float = 1e-3, dt_max: float = 0.1, dt_floor: float = 1e-4) -> torch.Tensor:
    dt = torch.exp(torch.rand(n) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
                   + torch.log(torch.tensor(dt_min))).clamp(min=dt_floor)
    return inverse_softplus(dt)


def acc_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Accumulation dtype for recurrent scans: float32, or float64 when the model runs in float64."""
    return torch.float64 if dtype == torch.float64 else torch.float32
