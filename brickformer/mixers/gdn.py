"""Gated DeltaNet mixer (Yang et al., 2024), following the ``fla`` layer layout.

Backends: ``'torch'`` (pure-PyTorch chunked WY form), ``'fla'`` (Triton kernels from
``flash-linear-attention``), ``'auto'`` (``fla`` on CUDA when installed).
Single-token decoding always uses the cheap recurrent update.

Recurrence per head with state ``S [K, V]``::

    S = S * exp(g_t)
    S = S + beta_t * k_t (v_t - S^T k_t)^T
    o_t = S^T (q_t / sqrt(K))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..norms import RMSNormGated
from ..registry import MIXERS
from .base import CausalConv1d, Mixer, acc_dtype, init_dt_bias


@dataclass
class GDNState:
    conv: torch.Tensor | None  # [B, C, conv_size - 1]
    recurrent: torch.Tensor  # [B, H, K, V] float32 (float64 for float64 models)


def gated_delta_rule_recurrent(q, k, v, g, beta, scale=None, initial_state=None):
    """Sequential reference. q, k: [B, T, H, K]; v: [B, T, H, V]; g, beta: [B, T, H]."""
    bsz, t, h, dk = q.shape
    scale = dk ** -0.5 if scale is None else scale
    s = q.new_zeros(bsz, h, dk, v.shape[-1]) if initial_state is None else initial_state
    out = []
    for i in range(t):
        s = s * torch.exp(g[:, i])[..., None, None]
        kv_mem = torch.einsum("bhkv,bhk->bhv", s, k[:, i])
        delta = (v[:, i] - kv_mem) * beta[:, i, :, None]
        s = s + k[:, i, :, :, None] * delta[:, :, None, :]
        out.append(torch.einsum("bhkv,bhk->bhv", s, q[:, i] * scale))
    return torch.stack(out, dim=1), s


def gated_delta_rule_chunked(q, k, v, g, beta, scale=None, initial_state=None, chunk_size: int = 64):
    """Chunked (WY representation) pure-PyTorch implementation; same contract as the recurrent one."""
    bsz, t, h, dk = q.shape
    dv = v.shape[-1]
    scale = dk ** -0.5 if scale is None else scale
    acc = acc_dtype(q.dtype)
    q, k, v, g, beta = (z.transpose(1, 2).to(acc) for z in (q, k, v, g, beta))  # [B, H, T, ...]
    pad = (-t) % chunk_size
    if pad:
        q, k, v = (F.pad(z, (0, 0, 0, pad)) for z in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (t + pad) // chunk_size
    q = q * scale
    k_beta = k * beta[..., None]
    v = v * beta[..., None]
    q, k, v, k_beta = (z.reshape(bsz, h, n, chunk_size, -1) for z in (q, k, v, k_beta))
    g = g.reshape(bsz, h, n, chunk_size).cumsum(-1)

    eye = torch.eye(chunk_size, device=q.device, dtype=acc)
    upper_incl = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
    decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()  # [B,H,N,C,C]
    # WY representation: T = (I + tril(diag(beta) K K^T * decay, -1))^{-1}
    m = ((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(upper_incl, 0)
    attn = torch.linalg.solve_triangular(eye + m, eye.expand_as(m), upper=False, unitriangular=True)
    u = attn @ v
    w = attn @ (k_beta * g.exp()[..., None])

    s = q.new_zeros(bsz, h, dk, dv) if initial_state is None else initial_state.to(acc)
    o = torch.empty_like(u)
    upper_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for i in range(n):
        q_i, k_i, u_i, g_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i]
        a_i = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill(upper_strict, 0)
        v_new = u_i - w[:, :, i] @ s
        o[:, :, i] = (q_i * g_i.exp()[..., None]) @ s + a_i @ v_new
        k_decay = k_i * (g_i[..., -1:] - g_i).exp()[..., None]
        s = s * g_i[..., -1, None, None].exp() + k_decay.transpose(-1, -2) @ v_new
    o = o.reshape(bsz, h, n * chunk_size, dv)[:, :, :t].transpose(1, 2)
    return o, s


def _fla_chunk():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as e:
        raise ImportError("GatedDeltaNet(backend='fla') requires `pip install flash-linear-attention` "
                          "(or use backend='torch')") from e
    return chunk_gated_delta_rule


@MIXERS.register_module()
class GatedDeltaNet(Mixer):
    def __init__(self, dim: int, heads: int = 8, head_dim: int | None = None, expand_v: float = 1.0,
                 use_short_conv: bool = True, conv_size: int = 4, use_gate: bool = True,
                 chunk_size: int = 64, norm_eps: float = 1e-5, backend: str = "auto"):
        super().__init__()
        if backend not in ("auto", "torch", "fla"):
            raise ValueError(f"GatedDeltaNet backend must be auto | torch | fla, got {backend!r}")
        if backend == "fla":
            _fla_chunk()
        self.dim, self.heads, self.backend, self.chunk_size = dim, heads, backend, chunk_size
        self.head_k = head_dim or dim // heads
        self.head_v = int(self.head_k * expand_v)
        self.key_dim, self.value_dim = heads * self.head_k, heads * self.head_v
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.a_proj = nn.Linear(dim, heads, bias=False)
        self.b_proj = nn.Linear(dim, heads, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(init_dt_bias(heads))
        self.conv = CausalConv1d(2 * self.key_dim + self.value_dim, conv_size, bias=False) if use_short_conv else None
        self.g_proj = nn.Linear(dim, self.value_dim, bias=False) if use_gate else None
        self.o_norm = RMSNormGated(self.head_v, norm_eps)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

    def init_state(self, batch_size, max_len=None, device=None, dtype=None) -> GDNState:
        conv = self.conv.init_state(batch_size, device, dtype) if self.conv is not None else None
        rec = torch.zeros(batch_size, self.heads, self.head_k, self.head_v, device=device, dtype=acc_dtype(dtype))
        return GDNState(conv, rec)

    def _use_fla(self, x) -> bool:
        if self.backend == "torch":
            return False
        if self.backend == "fla":
            return True
        try:
            _fla_chunk()
        except ImportError:
            return False
        return x.is_cuda

    def forward(self, x, state: GDNState | None = None, mask=None, position_ids=None):
        bsz, t, _ = x.shape
        if mask is not None:
            x = x * mask[..., None].to(x.dtype)
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        conv_state = None
        if self.conv is not None:
            qkv, conv_state = self.conv(qkv, state.conv if state is not None else None)
        else:
            qkv = F.silu(qkv)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        acc = acc_dtype(x.dtype)
        q = F.normalize(q.view(bsz, t, self.heads, self.head_k).to(acc), dim=-1, eps=1e-6)
        k = F.normalize(k.view(bsz, t, self.heads, self.head_k).to(acc), dim=-1, eps=1e-6)
        v = v.view(bsz, t, self.heads, self.head_v).to(acc)
        beta = torch.sigmoid(self.b_proj(x).to(acc))
        g = -self.A_log.to(acc).exp() * F.softplus(self.a_proj(x).to(acc) + self.dt_bias.to(acc))
        if mask is not None:
            beta = beta * mask[..., None].to(beta.dtype)
        init = state.recurrent if state is not None else None

        if t == 1:
            o, rec = gated_delta_rule_recurrent(q, k, v, g, beta, initial_state=init)
        elif self._use_fla(x):
            dt = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float32
            o, rec = _fla_chunk()(q.to(dt), k.to(dt), v.to(dt), g, beta.to(dt), scale=self.head_k ** -0.5,
                                  initial_state=init, output_final_state=state is not None)
        else:
            o, rec = gated_delta_rule_chunked(q, k, v, g, beta, initial_state=init, chunk_size=self.chunk_size)

        gate = self.g_proj(x).view(bsz, t, self.heads, self.head_v) if self.g_proj is not None else None
        o = self.o_norm(o.to(x.dtype), gate).reshape(bsz, t, self.value_dim)
        new_state = GDNState(conv_state, rec.to(acc)) if state is not None else None
        return self.o_proj(o), new_state

    def extra_flops_per_token(self, seq_len: int) -> float:
        # decay + delta read + rank-1 write + readout on a [K, V] state per head, plus intra-chunk term
        return 8.0 * self.heads * self.head_k * self.head_v + 2.0 * self.heads * self.chunk_size * (
            self.head_k + self.head_v)
