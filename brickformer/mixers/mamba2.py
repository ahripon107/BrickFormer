"""Mamba2 (SSD) mixer.

Backends: ``'torch'`` (pure-PyTorch chunked SSD, always available), ``'mamba_ssm'`` (Triton
kernel from the ``mamba_ssm`` package), or ``'auto'`` (``mamba_ssm`` on CUDA when installed).
Single-token decoding always uses the cheap recurrent update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..norms import RMSNormGated
from ..registry import MIXERS
from .base import CausalConv1d, Mixer, acc_dtype, init_dt_bias


@dataclass
class Mamba2State:
    conv: torch.Tensor  # [B, conv_dim, d_conv - 1]
    ssm: torch.Tensor  # [B, H, P, N] float32 (float64 for float64 models)


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum: out[..., i, j] = sum(x[..., j+1:i+1]) for j <= i else -inf."""
    t = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=t)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd_chunked(x, a, b, c, chunk_size: int = 64, initial_state=None):
    """Chunked SSD scan (pure PyTorch; run it in float32 or float64).

    x: [B, T, H, P] (already multiplied by dt); a: [B, T, H] (log decay = dt * A);
    b, c: [B, T, H, N]; initial_state: [B, H, P, N]. Returns ``(y [B,T,H,P], final_state)``.
    """
    t = x.shape[1]
    pad = (-t) % chunk_size
    if pad:  # zero input + zero log-decay leaves the state untouched
        x, b, c = (F.pad(z, (0, 0, 0, 0, 0, pad)) for z in (x, b, c))
        a = F.pad(a, (0, 0, 0, pad))
    x, a, b, c = (rearrange(z, "b (c l) ... -> b c l ...", l=chunk_size) for z in (x, a, b, c))
    a = rearrange(a, "b c l h -> b h c l")
    a_cumsum = torch.cumsum(a, dim=-1)

    decay = torch.exp(_segsum(a))  # intra-chunk
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, decay, x)

    decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)  # chunk-final states
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_states, x)

    if initial_state is None:
        initial_state = torch.zeros_like(states[:, :1])
    else:
        initial_state = initial_state[:, None]
    states = torch.cat([initial_state, states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(a_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, states, torch.exp(a_cumsum))
    y = rearrange(y_diag + y_off, "b c l h p -> b (c l) h p")
    return y[:, :t], final_state


def ssd_recurrent(x, a, b, c, initial_state=None):
    """Sequential reference of :func:`ssd_chunked` (same arguments)."""
    bsz, t, h, p = x.shape
    s = x.new_zeros(bsz, h, p, b.shape[-1]) if initial_state is None else initial_state
    ys = []
    for i in range(t):
        s = s * torch.exp(a[:, i])[..., None, None] + x[:, i, :, :, None] * b[:, i, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", s, c[:, i]))
    return torch.stack(ys, dim=1), s


def _mamba_ssm_scan():
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError as e:
        raise ImportError("Mamba2(backend='mamba_ssm') requires `pip install mamba-ssm` "
                          "(or use backend='torch')") from e
    return mamba_chunk_scan_combined


@MIXERS.register_module()
class Mamba2(Mixer):
    def __init__(self, dim: int, d_state: int = 128, d_conv: int = 4, expand: int = 2, headdim: int = 64,
                 ngroups: int = 1, chunk_size: int = 64, conv_bias: bool = True, bias: bool = False,
                 A_init_range: tuple[float, float] = (1.0, 16.0), dt_min: float = 1e-3, dt_max: float = 0.1,
                 norm_eps: float = 1e-5, backend: str = "auto"):
        super().__init__()
        if backend not in ("auto", "torch", "mamba_ssm"):
            raise ValueError(f"Mamba2 backend must be auto | torch | mamba_ssm, got {backend!r}")
        if backend == "mamba_ssm":
            _mamba_ssm_scan()
        self.dim, self.d_state, self.ngroups, self.headdim = dim, d_state, ngroups, headdim
        self.d_inner = expand * dim
        if self.d_inner % headdim:
            raise ValueError("expand * dim must be divisible by headdim")
        self.nheads = self.d_inner // headdim
        if self.nheads % ngroups:
            raise ValueError("nheads must be divisible by ngroups")
        self.chunk_size, self.backend = chunk_size, backend
        self.conv_dim = self.d_inner + 2 * ngroups * d_state
        self.in_proj = nn.Linear(dim, 2 * self.d_inner + 2 * ngroups * d_state + self.nheads, bias=bias)
        self.conv = CausalConv1d(self.conv_dim, d_conv, bias=conv_bias)
        self.dt_bias = nn.Parameter(init_dt_bias(self.nheads, dt_min, dt_max))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(*A_init_range)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(self.d_inner, norm_eps, group_size=self.d_inner // ngroups)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)

    def init_state(self, batch_size, max_len=None, device=None, dtype=None) -> Mamba2State:
        return Mamba2State(self.conv.init_state(batch_size, device, dtype),
                           torch.zeros(batch_size, self.nheads, self.headdim, self.d_state,
                                       device=device, dtype=acc_dtype(dtype)))

    def _use_kernel(self, x) -> bool:
        if self.backend == "torch":
            return False
        if self.backend == "mamba_ssm":
            return True
        try:
            _mamba_ssm_scan()
        except ImportError:
            return False
        return x.is_cuda

    def forward(self, x, state: Mamba2State | None = None, mask=None, position_ids=None):
        bsz, t, _ = x.shape
        if mask is not None:
            x = x * mask[..., None].to(x.dtype)
        z, xbc, dt = self.in_proj(x).split([self.d_inner, self.conv_dim, self.nheads], dim=-1)
        xbc, conv_state = self.conv(xbc, state.conv if state is not None else None)
        xs, bm, cm = xbc.split([self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        if mask is not None:
            xs = xs * mask[..., None].to(xs.dtype)
        acc = acc_dtype(x.dtype)
        dt = F.softplus(dt.to(acc) + self.dt_bias.to(acc))  # [B, T, H]
        A = -torch.exp(self.A_log.to(acc))
        xs = xs.view(bsz, t, self.nheads, self.headdim)
        bm = bm.view(bsz, t, self.ngroups, self.d_state)
        cm = cm.view(bsz, t, self.ngroups, self.d_state)
        init = state.ssm if state is not None else None

        if t > 1 and self._use_kernel(x):
            y, ssm = _mamba_ssm_scan()(xs, dt.to(xs.dtype), A, bm, cm, self.chunk_size,
                                       initial_states=init, return_final_states=True)
            y = y.to(acc)
        else:
            rep = self.nheads // self.ngroups
            bh = bm.to(acc).repeat_interleave(rep, dim=2)
            ch = cm.to(acc).repeat_interleave(rep, dim=2)
            xdt = xs.to(acc) * dt[..., None]
            adt = dt * A
            if t == 1:
                y, ssm = ssd_recurrent(xdt, adt, bh, ch, init)
            else:
                y, ssm = ssd_chunked(xdt, adt, bh, ch, self.chunk_size, init)
        y = y + xs.to(acc) * self.D.to(acc)[:, None]
        y = self.norm(y.reshape(bsz, t, self.d_inner).to(x.dtype), z)
        new_state = Mamba2State(conv_state, ssm.to(acc)) if state is not None else None
        return self.out_proj(y), new_state

    def extra_flops_per_token(self, seq_len: int) -> float:
        # state update + readout (~3 mults per state element) + intra-chunk quadratic term
        return 6.0 * self.nheads * self.headdim * self.d_state + 2.0 * self.nheads * self.chunk_size * (
            self.d_state + self.headdim)
