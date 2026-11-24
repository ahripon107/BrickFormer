"""A block = mixer slot + optional FFN slot + norms, wired by a residual style.

Residual styles (``n1``/``n2`` are the pre-mixer / pre-FFN norms):

* ``pre_norm``:  ``h = x + mixer(n1(x));        y = h + ffn(n2(h))``
* ``post_norm``: ``h = n1(x + mixer(x));        y = n2(h + ffn(h))``
* ``sandwich``:  ``h = x + n1_post(mixer(n1(x))); y = h + n2_post(ffn(n2(h)))``
* ``parallel``:  ``y = x + mixer(n1(x)) + ffn(n1(x))``  (GPT-J / PaLM style, one shared norm)
"""

from __future__ import annotations

from torch import nn

from .registry import BLOCKS, FFNS, MIXERS, NORMS

RESIDUAL_STYLES = ("pre_norm", "post_norm", "sandwich", "parallel")


@BLOCKS.register_module()
class Block(nn.Module):
    def __init__(self, dim: int, mixer: dict, ffn: dict | None = None, norm: dict | None = None,
                 residual: str = "pre_norm"):
        super().__init__()
        if residual not in RESIDUAL_STYLES:
            raise ValueError(f"residual must be one of {RESIDUAL_STYLES}, got {residual!r}")
        norm = norm or dict(type="RMSNorm")
        self.residual = residual
        self.mixer = MIXERS.build(mixer, dim=dim)
        self.ffn = FFNS.build(ffn, dim=dim) if ffn is not None else None
        self.norm1 = NORMS.build(norm, dim=dim)
        has_ffn_norm = self.ffn is not None and residual != "parallel"
        self.norm2 = NORMS.build(norm, dim=dim) if has_ffn_norm else None
        if residual == "sandwich":
            self.norm1_post = NORMS.build(norm, dim=dim)
            self.norm2_post = NORMS.build(norm, dim=dim) if self.ffn is not None else None

    def init_state(self, batch_size, max_len, device=None, dtype=None):
        return self.mixer.init_state(batch_size, max_len, device=device, dtype=dtype)

    def forward(self, x, state=None, mask=None, position_ids=None):
        r = self.residual
        if r == "pre_norm":
            y, state = self.mixer(self.norm1(x), state, mask, position_ids)
            h = x + y
            out = h + self.ffn(self.norm2(h)) if self.ffn is not None else h
        elif r == "post_norm":
            y, state = self.mixer(x, state, mask, position_ids)
            h = self.norm1(x + y)
            out = self.norm2(h + self.ffn(h)) if self.ffn is not None else h
        elif r == "sandwich":
            y, state = self.mixer(self.norm1(x), state, mask, position_ids)
            h = x + self.norm1_post(y)
            out = h + self.norm2_post(self.ffn(self.norm2(h))) if self.ffn is not None else h
        else:  # parallel
            n = self.norm1(x)
            y, state = self.mixer(n, state, mask, position_ids)
            out = x + y + (self.ffn(n) if self.ffn is not None else 0)
        return out, state
