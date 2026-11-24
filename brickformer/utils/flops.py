"""Parameter and FLOP accounting, grouped by block type.

FLOPs are forward-pass estimates: ``2 * active linear params`` per token plus each mixer's
``extra_flops_per_token`` (attention scores, recurrent state updates). Multiply by ~3 for a
training step. Non-matmul work (norms, activations, softmax) is ignored.
"""

from __future__ import annotations

from collections import defaultdict

from torch import nn

from ..ffns.moe import MoE


def _linear_params(module: nn.Module) -> int:
    return sum(m.weight.numel() for m in module.modules() if isinstance(m, nn.Linear))


def _active_ffn_params(ffn: nn.Module | None) -> int:
    if ffn is None:
        return 0
    if isinstance(ffn, MoE):
        return ffn.active_params()
    return _linear_params(ffn)


def count_params(model) -> dict:
    """Total / active parameters per block type, plus embeddings and head."""
    rows: dict = defaultdict(lambda: dict(layers=0, mixer=0, ffn=0, norm=0, total=0, active=0))
    for name, block in zip(model.layer_types, model.blocks):
        r = rows[name]
        mixer = sum(p.numel() for p in block.mixer.parameters())
        ffn = sum(p.numel() for p in block.ffn.parameters()) if block.ffn is not None else 0
        total = sum(p.numel() for p in block.parameters())
        r["layers"] += 1
        r["mixer"] += mixer
        r["ffn"] += ffn
        r["norm"] += total - mixer - ffn
        r["total"] += total
        r["active"] += total - ffn + (_active_ffn_params(block.ffn) if block.ffn is not None else 0)
    out = dict(rows)
    emb = model.embed.weight.numel()
    tied = model.lm_head.weight is model.embed.weight
    out["embed"] = dict(total=emb, active=emb)
    head = 0 if tied else model.lm_head.weight.numel()
    out["lm_head"] = dict(total=head, active=head)
    out["total"] = sum(p.numel() for p in model.parameters())
    return out


def estimate_flops(model, seq_len: int) -> dict:
    """Forward FLOPs per token, per block type and in total, for a sequence of ``seq_len``."""
    per_type: dict = defaultdict(float)
    for name, block in zip(model.layer_types, model.blocks):
        linear = _linear_params(block.mixer) + _active_ffn_params(block.ffn)
        per_type[name] += 2.0 * linear + block.mixer.extra_flops_per_token(seq_len)
    head = 2.0 * model.lm_head.weight.numel()
    total = sum(per_type.values()) + head
    return dict(per_type=dict(per_type), lm_head=head, per_token=total, per_sequence=total * seq_len,
                train_per_token=3 * total)


def summary(model, seq_len: int = 2048) -> str:
    p, f = count_params(model), estimate_flops(model, seq_len)
    lines = [f"{'block':<10}{'layers':>7}{'params':>14}{'active':>14}{'GFLOP/tok':>12}"]
    for name in dict.fromkeys(model.layer_types):
        r = p[name]
        lines.append(f"{name:<10}{r['layers']:>7}{r['total']:>14,}{r['active']:>14,}{f['per_type'][name] / 1e9:>12.4f}")
    lines.append(f"{'embed':<10}{'':>7}{p['embed']['total']:>14,}")
    lines.append(f"{'lm_head':<10}{'':>7}{p['lm_head']['total']:>14,}{'':>14}{f['lm_head'] / 1e9:>12.4f}")
    lines.append(f"{'total':<10}{len(model.blocks):>7}{p['total']:>14,}{'':>14}{f['per_token'] / 1e9:>12.4f}"
                 f"   (seq_len={seq_len}, forward)")
    return "\n".join(lines)
