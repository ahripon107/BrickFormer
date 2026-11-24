"""Load / dump configs (Python, YAML, JSON) and save / load checkpoints."""

from __future__ import annotations

import ast
import copy
import json
import runpy
from pathlib import Path
from typing import Any

import torch

from .registry import build_model


def load_config(path: str | Path) -> dict:
    """Python configs: all public top-level variables (mmdet style; usually ``model = dict(...)``).
    YAML / JSON: the parsed mapping."""
    path = Path(path)
    if path.suffix == ".py":
        ns = runpy.run_path(str(path))
        return {k: v for k, v in ns.items() if not k.startswith("_") and not callable(v)
                and not isinstance(v, type(json))}
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def dump_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    cfg = _to_plain(copy.deepcopy(cfg))
    if path.suffix in (".yaml", ".yml"):
        import yaml

        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    elif path.suffix == ".json":
        path.write_text(json.dumps(cfg, indent=2))
    elif path.suffix == ".py":
        if "type" in cfg:  # a bare model config
            path.write_text(f"model = {cfg!r}\n")
        else:  # an experiment config: one variable per section
            path.write_text("".join(f"{k} = {v!r}\n" for k, v in cfg.items()))
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        return text


def apply_overrides(cfg: dict, options: list[str] | dict | None) -> dict:
    """mmdet-style overrides: ``["train.learning_rate=1e-3", "model.block_types.attn.mixer.heads=8"]``.
    Values are parsed as Python literals when possible (``8``, ``1e-3``, ``[1, 2]``, ``'x'``, ``True``),
    otherwise kept as strings. Integer path segments index into lists. Returns a new dict."""
    cfg = copy.deepcopy(cfg)
    items = options.items() if isinstance(options, dict) else [o.split("=", 1) for o in options or []]
    for item in items:
        if len(item) != 2:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, value = item
        value = _parse_value(value) if isinstance(value, str) else value
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            if isinstance(node, list):
                node = node[int(part)]
            else:
                node = node.setdefault(part, {})
        if isinstance(node, list):
            node[int(leaf)] = value
        else:
            node[leaf] = value
    return cfg


def build_from_file(path: str | Path):
    cfg = load_config(path)
    return build_model(cfg["model"] if "model" in cfg else cfg)


def save_pretrained(model, directory: str | Path) -> None:
    """Writes ``config.json`` (the model's build config) and ``model.pt`` (state_dict)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dump_config(model.config, directory / "config.json")
    torch.save(model.state_dict(), directory / "model.pt")


def load_pretrained(directory: str | Path, map_location: str | torch.device = "cpu", strict: bool = True):
    directory = Path(directory)
    model = build_model(load_config(directory / "config.json"))
    model.load_state_dict(torch.load(directory / "model.pt", map_location=map_location), strict=strict)
    return model
