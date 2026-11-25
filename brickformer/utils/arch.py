"""Compare two model architectures (configs or built models) module by module."""

from __future__ import annotations

import re
from collections import defaultdict

import torch
from torch import nn

# attributes that do not change what a model computes or which weights it holds, or that only
# repeat parameter shapes (reported separately)
_IGNORED_ATTRS = {"training", "init_std", "grad_checkpointing", "layer_types", "config",
                  "in_features", "out_features", "num_embeddings", "embedding_dim"}
_PLAIN = (int, float, bool, str, type(None), dict, tuple)


def _as_model(model_or_cfg) -> nn.Module:
    if isinstance(model_or_cfg, nn.Module):
        return model_or_cfg
    from ..registry import build_model

    with torch.device("meta"):  # no memory, no init cost
        return build_model(model_or_cfg)


def _signature(model: nn.Module) -> tuple[dict, dict, bool]:
    modules = {}
    for name, mod in model.named_modules():
        fwd = type(mod).forward
        # subclasses that only change defaults (MHAttention -> GQAttention) share a forward
        entry = {"type": f"{fwd.__qualname__.rsplit('.', 1)[0]}"}
        for k, v in vars(mod).items():
            if not k.startswith("_") and k not in _IGNORED_ATTRS and isinstance(v, _PLAIN):
                entry[k] = v
        modules[name] = entry
    params = {k: tuple(p.shape) for k, p in model.named_parameters()}  # tied weights appear once
    lm_head, embed = getattr(model, "lm_head", None), getattr(model, "embed", None)
    tied = lm_head is not None and embed is not None and lm_head.weight is embed.weight
    return modules, params, tied


def _ranges(nums: list[int]) -> str:
    nums, out, start = sorted(nums), [], None
    for i, n in enumerate(nums):
        start = n if start is None else start
        if i + 1 == len(nums) or nums[i + 1] != n + 1:
            out.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ",".join(out)


def _collapse(diffs: list[tuple[str, str]]) -> list[str]:
    """Group identical differences across layers: ``blocks.[0-27].mixer.head_dim: ...``."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    order = []
    for path, msg in diffs:
        m = re.match(r"^blocks\.(\d+)(.*)$", path)
        key = ("blocks.{}" + m.group(2), msg) if m else (path, msg)
        if key not in groups:
            order.append(key)
        groups[key].append(int(m.group(1)) if m else -1)
    lines = []
    for key in order:
        path, msg = key
        idx = [i for i in groups[key] if i >= 0]
        path = path.format(f"[{_ranges(idx)}]") if idx else path
        lines.append(f"{path or '<model>'}: {msg}")
    return lines


def architecture_diff(a, b, a_name: str = "config", b_name: str = "pretrained") -> list[str]:
    """Differences between two architectures, each a model config dict or a built model.

    Returns an empty list when both compute the same function with the same parameter shapes.
    Compares module types, hyperparameters (heads, head_dim, window, RoPE theta / scaling, norm eps,
    residual style...), parameter names and shapes, and embedding tying. Training-only settings
    (``grad_checkpointing``, ``init_std``) and block-type names are ignored.
    """
    ma, pa, ta = _signature(_as_model(a))
    mb, pb, tb = _signature(_as_model(b))
    diffs: list[tuple[str, str]] = []
    na = sum(1 for k in ma if re.fullmatch(r"blocks\.\d+", k))
    nb = sum(1 for k in mb if re.fullmatch(r"blocks\.\d+", k))
    if na != nb:
        diffs.append(("num_layers", f"{a_name}={na}, {b_name}={nb}"))
    if ta != tb:
        diffs.append(("tie_embeddings", f"{a_name}={ta}, {b_name}={tb}"))

    def only_in(name_set, other, label):
        for k in sorted(name_set - other, key=_natural):
            parent = k.rsplit(".", 1)[0] if "." in k else None
            if parent is None or parent in other:  # report the top-most missing module only
                diffs.append((k, f"only in {label}"))

    common = set(ma) & set(mb)
    only_in(set(ma), set(mb), a_name)
    only_in(set(mb), set(ma), b_name)
    for k in sorted(common, key=_natural):
        ea, eb = ma[k], mb[k]
        for attr in sorted(set(ea) | set(eb), key=lambda s: (s != "type", s)):
            path = f"{k}.{attr}" if k else attr
            if path in ma or path in mb:  # an optional submodule set to None: reported as a missing module
                continue
            va, vb = ea.get(attr, "<unset>"), eb.get(attr, "<unset>")
            if va != vb:
                diffs.append((path, f"{a_name}={va!r}, {b_name}={vb!r}"))
    for k in sorted(set(pa) | set(pb), key=_natural):
        owner = k.rsplit(".", 1)[0]
        if owner not in common or (k == "lm_head.weight" and ta != tb):
            continue  # already reported as a missing module / tying difference
        sa, sb = pa.get(k), pb.get(k)
        if sa != sb:
            fa = "absent" if sa is None else list(sa)
            fb = "absent" if sb is None else list(sb)
            diffs.append((k, f"parameter shape {a_name}={fa}, {b_name}={fb}"))
    return _collapse(diffs)


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


class ArchitectureMismatch(ValueError):
    """Raised when a model config does not match the architecture of the weights being loaded."""
