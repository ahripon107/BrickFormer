"""Multi-head Latent Attention (DeepSeek-V2 style).

KV is compressed into a ``kv_lora_rank`` latent plus a single shared RoPE key; only those are
cached, which is the point of MLA. The cache is decompressed on every step (the simple,
non-"absorbed" formulation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..norms import RMSNorm
from ..positional import build_rope
from ..registry import MIXERS
from .attention import sdpa
from .base import Mixer


@dataclass
class MLACache:
    c_kv: torch.Tensor  # [B, max_len, kv_lora_rank]
    k_pe: torch.Tensor  # [B, max_len, rope_head_dim]
    pos: int = 0

    def append(self, c_kv, k_pe):
        t = c_kv.shape[1]
        if self.pos + t > self.c_kv.shape[1]:
            raise RuntimeError(f"MLA cache overflow: {self.pos + t} > max_len {self.c_kv.shape[1]}")
        self.c_kv[:, self.pos:self.pos + t] = c_kv
        self.k_pe[:, self.pos:self.pos + t] = k_pe
        self.pos += t
        return self.c_kv[:, :self.pos], self.k_pe[:, :self.pos]


@MIXERS.register_module()
class MLA(Mixer):
    def __init__(self, dim: int, heads: int, kv_lora_rank: int = 512, q_lora_rank: int | None = None,
                 nope_head_dim: int = 128, rope_head_dim: int = 64, v_head_dim: int = 128,
                 rope=True, window: int | None = None, norm_eps: float = 1e-6):
        super().__init__()
        self.dim, self.heads, self.window = dim, heads, window
        self.kv_lora_rank, self.nope_head_dim, self.rope_head_dim, self.v_head_dim = (
            kv_lora_rank, nope_head_dim, rope_head_dim, v_head_dim)
        self.qk_head_dim = nope_head_dim + rope_head_dim
        if q_lora_rank:
            self.q_a_proj = nn.Linear(dim, q_lora_rank, bias=False)
            self.q_a_norm = RMSNorm(q_lora_rank, norm_eps)
            self.q_b_proj = nn.Linear(q_lora_rank, heads * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(dim, heads * self.qk_head_dim, bias=False)
        self.q_lora_rank = q_lora_rank
        self.kv_a_proj = nn.Linear(dim, kv_lora_rank + rope_head_dim, bias=False)
        self.kv_a_norm = RMSNorm(kv_lora_rank, norm_eps)
        self.kv_b_proj = nn.Linear(kv_lora_rank, heads * (nope_head_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(heads * v_head_dim, dim, bias=False)
        self.rope = build_rope(rope, rope_head_dim)

    def init_state(self, batch_size, max_len, device=None, dtype=None) -> MLACache:
        return MLACache(torch.zeros(batch_size, max_len, self.kv_lora_rank, device=device, dtype=dtype),
                        torch.zeros(batch_size, max_len, self.rope_head_dim, device=device, dtype=dtype))

    def forward(self, x, state: MLACache | None = None, mask=None, position_ids=None):
        b, t, _ = x.shape
        h = self.heads
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x))) if self.q_lora_rank else self.q_proj(x)
        q = q.view(b, t, h, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        c_kv, k_pe = self.kv_a_proj(x).split([self.kv_lora_rank, self.rope_head_dim], dim=-1)
        k_pe = k_pe[:, None]  # [B, 1, T, R] shared across heads
        offset = state.pos if state is not None else 0
        if self.rope is not None:
            if position_ids is None:
                position_ids = torch.arange(offset, offset + t, device=x.device).expand(b, t)
            cos, sin = self.rope.cos_sin(position_ids, q.dtype)
            q_pe, k_pe = self.rope.rotate(q_pe, cos, sin), self.rope.rotate(k_pe, cos, sin)
        k_pe = k_pe[:, 0]
        if state is not None:
            if mask is not None:
                raise NotImplementedError("key-padding masks are not supported together with a cache")
            c_kv, k_pe = state.append(c_kv, k_pe)
        tk = c_kv.shape[1]
        kv = self.kv_b_proj(self.kv_a_norm(c_kv)).view(b, tk, h, self.nope_head_dim + self.v_head_dim)
        k_nope, v = kv.transpose(1, 2).split([self.nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe[:, None].expand(b, h, tk, self.rope_head_dim)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        y = sdpa(q, k, v, offset, self.window, mask, scale=self.qk_head_dim ** -0.5)
        y = y.transpose(1, 2).reshape(b, t, h * self.v_head_dim)
        return self.o_proj(y), state

    def extra_flops_per_token(self, seq_len: int) -> float:
        ctx = seq_len / 2 if self.window is None else min(seq_len / 2, self.window)
        return 2.0 * self.heads * (self.qk_head_dim + self.v_head_dim) * ctx
