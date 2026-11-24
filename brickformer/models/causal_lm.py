from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..registry import BLOCKS, MODELS, NORMS

# Single-character shorthand used by compact pattern strings like "GGG*".
DEFAULT_PATTERN_CHARS = {"*": "attn", "A": "attn", "W": "swa", "L": "mla", "M": "mamba", "G": "gdn"}


def expand_pattern(pattern: str | list[str], block_types: dict, num_layers: int | None = None,
                   pattern_chars: dict | None = None) -> list[str]:
    """Turn ``pattern`` into a list of ``block_types`` keys.

    A string is read one character at a time: a character that is itself a key of
    ``block_types`` maps to that key, otherwise ``pattern_chars`` (merged over
    :data:`DEFAULT_PATTERN_CHARS`) is used. ``-`` and spaces are ignored. If ``num_layers``
    is given, the pattern is tiled to that length (it must divide it).
    """
    if isinstance(pattern, str):
        chars = {**DEFAULT_PATTERN_CHARS, **(pattern_chars or {})}
        names = []
        for ch in pattern:
            if ch in " -_|":
                continue
            name = ch if ch in block_types else chars.get(ch)
            if name is None:
                raise KeyError(f"pattern character {ch!r} maps to no block type; pass pattern_chars")
            names.append(name)
    else:
        names = list(pattern)
    if num_layers is not None and len(names) != num_layers:
        if num_layers % len(names):
            raise ValueError(f"pattern of length {len(names)} cannot be tiled to {num_layers} layers")
        names = names * (num_layers // len(names))
    unknown = sorted(set(names) - set(block_types))
    if unknown:
        raise KeyError(f"pattern uses undefined block types {unknown}; defined: {sorted(block_types)}")
    return names


@dataclass
class ModelCache:
    """Per-layer mixer states plus the number of tokens processed so far."""

    states: list[Any]
    seq_len: int = 0


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    cache: ModelCache | None = None
    extras: dict = field(default_factory=dict)


@MODELS.register_module()
class CausalLM(nn.Module):
    """Embeddings -> stack of blocks (chosen by ``pattern``) -> final norm -> LM head.

    Each entry of ``block_types`` is a ``Block`` config without ``dim``; ``norm`` and
    ``residual`` given at the model level are defaults that a block type may override.
    """

    def __init__(self, vocab_size: int, dim: int, pattern: str | list[str], block_types: dict,
                 num_layers: int | None = None, pattern_chars: dict | None = None,
                 norm: dict | None = None, residual: str = "pre_norm", final_norm: bool = True,
                 tie_embeddings: bool = False, init_std: float = 0.02, grad_checkpointing: bool = False):
        super().__init__()
        self.config = copy.deepcopy(dict(
            type="CausalLM", vocab_size=vocab_size, dim=dim, pattern=pattern, block_types=block_types,
            num_layers=num_layers, pattern_chars=pattern_chars, norm=norm, residual=residual,
            final_norm=final_norm, tie_embeddings=tie_embeddings, init_std=init_std,
            grad_checkpointing=grad_checkpointing))
        norm = norm or dict(type="RMSNorm")
        self.vocab_size, self.dim, self.init_std = vocab_size, dim, init_std
        self.grad_checkpointing = grad_checkpointing
        self.layer_types = expand_pattern(pattern, block_types, num_layers, pattern_chars)

        self.embed = nn.Embedding(vocab_size, dim)
        blocks = []
        for name in self.layer_types:
            cfg = dict(type="Block", norm=norm, residual=residual)
            cfg.update(copy.deepcopy(block_types[name]))
            blocks.append(BLOCKS.build(cfg, dim=dim))
        self.blocks = nn.ModuleList(blocks)
        self.norm = NORMS.build(norm, dim=dim) if final_norm else nn.Identity()
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._aux_modules = [m for m in self.modules() if hasattr(m, "aux_loss")]

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.init_std)

    def init_cache(self, batch_size: int, max_len: int, device=None, dtype=None) -> ModelCache:
        p = next(self.parameters())
        device, dtype = device or p.device, dtype or p.dtype
        return ModelCache([b.init_state(batch_size, max_len, device, dtype) for b in self.blocks])

    def forward(self, input_ids: torch.Tensor, cache: ModelCache | None = None,
                attention_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                labels: torch.Tensor | None = None) -> CausalLMOutput:
        """``labels`` are aligned with ``input_ids`` (shifted internally); ``-100`` is ignored.
        ``attention_mask`` is a ``[B, T]`` key-padding mask (training / no-cache only)."""
        b, t = input_ids.shape
        offset = cache.seq_len if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(offset, offset + t, device=input_ids.device).expand(b, t)
        h = self.embed(input_ids)
        use_ckpt = self.grad_checkpointing and self.training and cache is None
        for i, block in enumerate(self.blocks):
            state = cache.states[i] if cache is not None else None
            if use_ckpt:
                h, state = checkpoint(block, h, state, attention_mask, position_ids, use_reentrant=False)
            else:
                h, state = block(h, state, attention_mask, position_ids)
            if cache is not None:
                cache.states[i] = state
        if cache is not None:
            cache.seq_len += t
        logits = self.lm_head(self.norm(h))

        aux = None
        if self._aux_modules:
            losses = [m.aux_loss for m in self._aux_modules if m.aux_loss is not None]
            aux = torch.stack(losses).sum() if losses else None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            if aux is not None and self.training:
                loss = loss + aux
        return CausalLMOutput(logits=logits, loss=loss, aux_loss=aux, cache=cache)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        from ..generation import generate

        return generate(self, input_ids, max_new_tokens, **kwargs)

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n
