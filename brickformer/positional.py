"""Rotary position embeddings (HF/Llama ``rotate_half`` convention)."""

from __future__ import annotations

import math

import torch
from torch import nn

from .registry import POS_EMBEDS


def _rope_inv_freq(head_dim: int, theta: float, scaling: dict | None) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if not scaling:
        return inv_freq
    kind = scaling.get("type", "linear")
    factor = float(scaling.get("factor", 1.0))
    if kind == "linear":
        return inv_freq / factor
    if kind == "ntk":  # static NTK-aware: stretch the base
        theta = theta * factor ** (head_dim / (head_dim - 2))
        return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if kind == "llama3":
        low = float(scaling.get("low_freq_factor", 1.0))
        high = float(scaling.get("high_freq_factor", 4.0))
        old_ctx = float(scaling.get("original_max_position_embeddings", 8192))
        low_wavelen, high_wavelen = old_ctx / low, old_ctx / high
        wavelen = 2 * math.pi / inv_freq
        scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
        smooth = (old_ctx / wavelen - low) / (high - low)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        is_medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
        return torch.where(is_medium, smoothed, scaled)
    raise ValueError(f"Unknown RoPE scaling type {kind!r} (expected linear | ntk | llama3)")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@POS_EMBEDS.register_module()
class RoPE(nn.Module):
    """Applies rotary embeddings to tensors shaped ``[B, H, T, head_dim]``.

    ``scaling``: ``None`` or ``dict(type='linear'|'ntk'|'llama3', factor=..., ...)``.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0, scaling: dict | None = None):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        self.head_dim, self.theta, self.scaling = head_dim, theta, scaling
        self.register_buffer("inv_freq", _rope_inv_freq(head_dim, theta, scaling), persistent=False)

    def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float()[..., None] * self.inv_freq.float()  # [B, T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)[:, None]  # [B, 1, T, D]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return x * cos + rotate_half(x) * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        cos, sin = self.cos_sin(position_ids, q.dtype)
        return self.rotate(q, cos, sin), self.rotate(k, cos, sin)


@POS_EMBEDS.register_module(name="NoPE")
class NoPE(nn.Module):
    def __init__(self, head_dim: int | None = None):
        super().__init__()

    def forward(self, q, k, position_ids):
        return q, k


def build_rope(rope, head_dim: int) -> nn.Module | None:
    """``True`` -> default RoPE, ``False``/``None`` -> no positional encoding, dict -> registry build."""
    if rope is None or rope is False:
        return None
    if rope is True:
        rope = dict(type="RoPE")
    return POS_EMBEDS.build(rope, head_dim=head_dim)
