"""Softmax attention mixers: MHA, GQA, sliding-window. Uses ``F.scaled_dot_product_attention``,
which dispatches to FlashAttention / memory-efficient kernels when the inputs allow it."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..norms import RMSNorm
from ..positional import build_rope
from ..registry import MIXERS
from .base import Mixer


@dataclass
class KVCache:
    k: torch.Tensor  # [B, H_kv, max_len, D]
    v: torch.Tensor
    pos: int = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[2]
        if self.pos + t > self.k.shape[2]:
            raise RuntimeError(f"KV cache overflow: {self.pos + t} > max_len {self.k.shape[2]}")
        self.k[:, :, self.pos:self.pos + t] = k
        self.v[:, :, self.pos:self.pos + t] = v
        self.pos += t
        return self.k[:, :, :self.pos], self.v[:, :, :self.pos]


def attention_mask(q_len: int, k_len: int, offset: int, window: int | None,
                   key_padding: torch.Tensor | None, device) -> torch.Tensor | None:
    """Boolean mask (True = attend) for causal (+ optional sliding window, key padding) attention.
    Returns ``None`` when plain ``is_causal`` or no mask suffices (callers handle those)."""
    q_idx = torch.arange(offset, offset + q_len, device=device)[:, None]
    k_idx = torch.arange(k_len, device=device)[None, :]
    m = k_idx <= q_idx
    if window is not None:
        m = m & (q_idx - k_idx < window)
    if key_padding is not None:
        # keep self-attention so fully padded query rows do not produce NaNs
        m = (m[None] & key_padding[:, None, :].bool()) | (k_idx == q_idx)[None]
        m = m[:, None]  # [B, 1, Tq, Tk]
    return m


def sdpa(q, k, v, offset: int, window: int | None, key_padding, scale: float | None = None):
    """q: [B, H, Tq, D]; k, v: [B, H, Tk, D] (already expanded to H heads)."""
    tq, tk = q.shape[2], k.shape[2]
    if window is None and key_padding is None:
        if tq == tk and offset == 0:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        if tq == 1:
            return F.scaled_dot_product_attention(q, k, v, scale=scale)
    mask = attention_mask(tq, tk, offset, window, key_padding, q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


@MIXERS.register_module()
class GQAttention(Mixer):
    """Grouped-query attention. ``kv_heads=None`` gives MHA, ``kv_heads=1`` gives MQA.

    Args:
        rope: ``True`` / ``False`` / a ``POS_EMBEDS`` config such as
            ``dict(type='RoPE', theta=5e5, scaling=dict(type='llama3', factor=8))``.
        window: sliding-window size (each query sees the last ``window`` positions incl. itself).
        qk_norm: apply RMSNorm over ``head_dim`` to queries and keys.
    """

    def __init__(self, dim: int, heads: int, kv_heads: int | None = None, head_dim: int | None = None,
                 rope=True, qk_norm: bool = False, window: int | None = None, bias: bool = False,
                 out_bias: bool = False, norm_eps: float = 1e-6):
        super().__init__()
        kv_heads = kv_heads or heads
        if heads % kv_heads:
            raise ValueError(f"heads ({heads}) must be divisible by kv_heads ({kv_heads})")
        self.dim, self.heads, self.kv_heads = dim, heads, kv_heads
        self.head_dim = head_dim or dim // heads
        self.window = window
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=out_bias)
        self.q_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else None
        self.rope = build_rope(rope, self.head_dim)

    def init_state(self, batch_size, max_len, device=None, dtype=None) -> KVCache:
        shape = (batch_size, self.kv_heads, max_len, self.head_dim)
        return KVCache(torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype))

    def forward(self, x, state: KVCache | None = None, mask=None, position_ids=None):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        offset = state.pos if state is not None else 0
        if self.rope is not None:
            if position_ids is None:
                position_ids = torch.arange(offset, offset + t, device=x.device).expand(b, t)
            q, k = self.rope(q, k, position_ids)
        if state is not None:
            if mask is not None:
                raise NotImplementedError("key-padding masks are not supported together with a cache")
            k, v = state.append(k, v)
        if self.kv_heads != self.heads:
            rep = self.heads // self.kv_heads
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = sdpa(q, k, v, offset, self.window, mask)
        y = y.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.o_proj(y), state

    def extra_flops_per_token(self, seq_len: int) -> float:
        ctx = seq_len / 2 if self.window is None else min(seq_len / 2, self.window)
        return 4.0 * self.heads * self.head_dim * ctx  # QK^T + AV


@MIXERS.register_module()
class MHAttention(GQAttention):
    def __init__(self, dim: int, heads: int, **kwargs):
        if kwargs.get("kv_heads") not in (None, heads):
            raise ValueError("MHAttention has kv_heads == heads; use GQAttention for grouped KV")
        kwargs.pop("kv_heads", None)
        super().__init__(dim, heads, kv_heads=heads, **kwargs)


@MIXERS.register_module()
class SlidingWindowAttention(GQAttention):
    def __init__(self, dim: int, heads: int, window: int = 4096, **kwargs):
        super().__init__(dim, heads, window=window, **kwargs)
