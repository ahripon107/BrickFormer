"""Build and run a whole experiment from one config::

    objective = 'pretrain'            # 'sft' | dict(type='dpo', ...) | dict(type='grpo', ...); default 'pretrain'
    init_from = 'runs/pretrain'       # optional: start from a BrickFormer checkpoint
    model = dict(type='CausalLM', vocab_size='auto', ...)   # optional with init_from
    tokenizer = dict(type='HFTokenizer', name='gpt2')        # optional with init_from
    data = dict(train=dict(path=..., ...), val=dict(...))    # dataset type defaults from the objective
    train = dict(type='HFTrainer', output_dir='runs/x', ...)

Typical chain: pretrain -> ``init_from`` the pretrain run for SFT -> ``init_from`` the SFT run for
DPO or GRPO.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .. import data as _data  # noqa: F401  (registers tokenizers and datasets)
from ..config import load_config
from ..registry import DATASETS, RUNNERS, TOKENIZERS, build_model
from . import hf_trainer as _hf_trainer  # noqa: F401  (registers runners)
from .objectives import Objective, build_objective


def resolve_vocab_size(model_cfg: dict, tokenizer, multiple_of: int = 64) -> dict:
    """``vocab_size='auto'`` (or missing) -> tokenizer vocab rounded up to ``multiple_of`` (faster
    matmuls; the extra rows are never targets). An explicit size must cover the tokenizer."""
    model_cfg = copy.deepcopy(model_cfg)
    vs = model_cfg.get("vocab_size", "auto")
    if vs in (None, "auto"):
        model_cfg["vocab_size"] = -(-tokenizer.vocab_size // multiple_of) * multiple_of
    elif vs < tokenizer.vocab_size:
        raise ValueError(f"model.vocab_size={vs} is smaller than the tokenizer's vocabulary ({tokenizer.vocab_size})")
    return model_cfg


def _resolve_init_from(cfg: dict) -> None:
    """Fill ``model`` / ``tokenizer`` from the ``init_from`` checkpoint when they are not given."""
    ckpt = Path(cfg["init_from"])
    if not (ckpt / "model.pt").is_file():
        raise FileNotFoundError(f"init_from={str(ckpt)!r} is not a BrickFormer checkpoint (no model.pt)")
    if "model" not in cfg:
        cfg["model"] = load_config(ckpt / "config.json")
    if "tokenizer" not in cfg:
        exp = ckpt / "experiment.json"
        if not exp.is_file():
            raise KeyError(f"no tokenizer section and {ckpt} has no experiment.json to take it from")
        cfg["tokenizer"] = load_config(exp)["tokenizer"]


def _dataset_cfg(data_cfg: dict, objective: Objective) -> dict:
    data_cfg = dict(data_cfg)
    data_cfg.setdefault("type", objective.default_dataset)
    served = getattr(DATASETS.get(data_cfg["type"]), "objectives", None)
    if served is not None and objective.name not in served:
        raise ValueError(f"dataset type {data_cfg['type']!r} is for objective(s) {sorted(served)}, but the "
                         f"objective is {objective.name!r}; use {objective.default_dataset!r} or omit 'type'")
    return data_cfg


@dataclass
class Experiment:
    config: dict  # fully resolved (model and tokenizer filled in); saved with every checkpoint
    objective: Objective
    model: Any
    tokenizer: Any
    train_dataset: Any
    eval_dataset: Any
    runner: Any

    def run(self):
        return self.runner.run(self.model, self.tokenizer, self.train_dataset, self.eval_dataset,
                               objective=self.objective, experiment_config=self.config)


def build_experiment(cfg: dict, build_data: bool = True) -> Experiment:
    cfg = copy.deepcopy(cfg)
    if cfg.get("init_from"):
        _resolve_init_from(cfg)
    missing = [s for s in ("model", "tokenizer", "data", "train") if s not in cfg]
    if missing:
        raise KeyError(f"experiment config is missing sections {missing} (model and tokenizer may come from "
                       f"init_from); needs model, tokenizer, data, train")
    cfg.setdefault("objective", "pretrain")
    objective = build_objective(cfg["objective"])
    tokenizer = TOKENIZERS.build(cfg["tokenizer"])
    cfg["model"] = resolve_vocab_size(cfg["model"], tokenizer, cfg.get("vocab_multiple_of", 64))

    torch.manual_seed(cfg["train"].get("seed", 42))  # model init happens before the Trainer seeds
    model = build_model(cfg["model"])
    if cfg.get("init_from"):
        state = torch.load(Path(cfg["init_from"]) / "model.pt", map_location="cpu")
        model.load_state_dict(state)

    unknown = set(cfg["data"]) - {"train", "val"}
    if unknown:
        raise KeyError(f"data section accepts 'train' and 'val', got {sorted(unknown)}")
    train_cfg = _dataset_cfg(cfg["data"]["train"], objective)
    val_cfg = _dataset_cfg(cfg["data"]["val"], objective) if cfg["data"].get("val") is not None else None
    train_ds = eval_ds = None
    if build_data:
        train_ds = DATASETS.build(train_cfg, tokenizer=tokenizer)
        eval_ds = DATASETS.build(val_cfg, tokenizer=tokenizer) if val_cfg is not None else None
    runner = RUNNERS.build(cfg["train"])
    return Experiment(cfg, objective, model, tokenizer, train_ds, eval_ds, runner)


def run_experiment(cfg: dict):
    return build_experiment(cfg).run()
