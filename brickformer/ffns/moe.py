"""Mixture-of-Experts FFN: top-k softmax router, SwiGLU experts, optional shared experts,
Switch-style load-balancing aux loss and optional router z-loss.

After each forward the module exposes ``self.aux_loss`` (already scaled by the coefficients);
``CausalLM`` sums these into ``output.aux_loss``. Experts are run with a simple per-expert
gather/scatter loop, which is fine for single-GPU research but not a grouped-GEMM kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import FFNS
from .dense import SwiGLU


@FFNS.register_module()
class MoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2, hidden: int | None = None,
                 shared_experts: int = 0, shared_hidden: int | None = None, norm_topk_prob: bool = True,
                 aux_loss_coef: float = 0.01, z_loss_coef: float = 0.0, init_std: float = 0.02):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("need 1 <= top_k <= num_experts")
        hidden = hidden or 4 * dim // top_k
        self.dim, self.num_experts, self.top_k, self.hidden = dim, num_experts, top_k, hidden
        self.norm_topk_prob, self.aux_loss_coef, self.z_loss_coef = norm_topk_prob, aux_loss_coef, z_loss_coef
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.w_gate = nn.Parameter(torch.randn(num_experts, dim, hidden) * init_std)
        self.w_up = nn.Parameter(torch.randn(num_experts, dim, hidden) * init_std)
        self.w_down = nn.Parameter(torch.randn(num_experts, hidden, dim) * init_std)
        self.shared = (SwiGLU(dim, (shared_hidden or hidden) * shared_experts) if shared_experts else None)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        n = x.shape[0]
        logits = self.router(x).float()
        probs = logits.softmax(-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_p = top_p / top_p.sum(-1, keepdim=True)

        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            tok, slot = torch.where(top_i == e)
            if tok.numel() == 0:
                continue
            xe = x[tok]
            h = F.silu(xe @ self.w_gate[e]) * (xe @ self.w_up[e])
            out.index_add_(0, tok, (h @ self.w_down[e]) * top_p[tok, slot, None].to(x.dtype))
        if self.shared is not None:
            out = out + self.shared(x)

        # Switch-Transformer balance loss: E * sum_e (fraction routed to e) * (mean prob of e)
        frac = torch.zeros(self.num_experts, device=x.device).scatter_add_(
            0, top_i.reshape(-1), torch.ones(n * self.top_k, device=x.device)) / (n * self.top_k)
        aux = self.aux_loss_coef * self.num_experts * (frac * probs.mean(0)).sum()
        if self.z_loss_coef:
            aux = aux + self.z_loss_coef * torch.logsumexp(logits, -1).pow(2).mean()
        self.aux_loss = aux
        return out.view(shape)

    def active_params(self) -> int:
        per_expert = 3 * self.dim * self.hidden
        shared = sum(p.numel() for p in self.shared.parameters()) if self.shared is not None else 0
        return self.top_k * per_expert + shared + self.router.weight.numel()
