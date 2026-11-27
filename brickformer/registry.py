"""A minimal mmengine-style registry.

Components register themselves under a string name and are instantiated from
config dicts of the form ``dict(type='Name', **kwargs)``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._modules: dict[str, type] = {}

    def register_module(self, name: str | None = None, force: bool = False) -> Callable[[type], type]:
        def _register(cls: type) -> type:
            key = name or cls.__name__
            if key in self._modules and not force:
                raise KeyError(f"'{key}' is already registered in {self.name}")
            self._modules[key] = cls
            return cls

        return _register

    def get(self, name: str) -> type:
        if name not in self._modules:
            raise KeyError(
                f"'{name}' is not registered in {self.name}. Available: {sorted(self._modules)}"
            )
        return self._modules[name]

    def build(self, cfg: dict | None, **default_args: Any) -> Any:
        """Instantiate ``cfg['type']`` with the remaining keys; ``default_args`` fill in missing keys."""
        if cfg is None:
            return None
        if not isinstance(cfg, dict) or "type" not in cfg:
            raise TypeError(f"{self.name}: config must be a dict with a 'type' key, got {cfg!r}")
        cfg = copy.deepcopy(cfg)
        cls = self.get(cfg.pop("type"))
        for k, v in default_args.items():
            cfg.setdefault(k, v)
        return cls(**cfg)

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __repr__(self) -> str:
        return f"Registry({self.name}, {sorted(self._modules)})"


MIXERS = Registry("mixers")
FFNS = Registry("ffns")
NORMS = Registry("norms")
POS_EMBEDS = Registry("pos_embeds")
BLOCKS = Registry("blocks")
MODELS = Registry("models")
TOKENIZERS = Registry("tokenizers")
DATASETS = Registry("datasets")
RUNNERS = Registry("runners")
OBJECTIVES = Registry("objectives")
REWARDS = Registry("rewards")
BENCHMARKS = Registry("benchmarks")


def build_model(cfg: dict) -> Any:
    return MODELS.build(cfg)
