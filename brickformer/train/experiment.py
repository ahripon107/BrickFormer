"""Build and run a whole experiment from one config::

    objective = 'pretrain'            # 'sft' | dict(type='dpo', ...) | dict(type='grpo', ...); default 'pretrain'
    init_from = 'runs/pretrain'       # optional: start from a BrickFormer checkpoint
    pretrained = 'Qwen/Qwen3-0.6B'    # optional: start from HF weights (Llama / Mistral / Qwen2 / Qwen3)
    model = dict(type='CausalLM', vocab_size='auto', ...)   # optional with init_from / pretrained
    tokenizer = dict(type='HFTokenizer', name='gpt2')        # optional with init_from / pretrained
    data = dict(train=dict(path=..., ...), val=dict(...))    # dataset type defaults from the objective
    train = dict(type='HFTrainer', output_dir='runs/x', ...)

Typical chain: pretrain -> ``init_from`` the pretrain run for SFT -> ``init_from`` the SFT run for
DPO or GRPO.

``pretrained`` (a Hub id / local HF directory, or ``dict(path=..., revision=None, strict=True)``):

* with a full ``model`` section, the weights are loaded only if that config describes exactly the
  checkpoint's architecture; otherwise ``ArchitectureMismatch`` lists every difference.
  ``strict=False`` only requires parameter names and shapes to match (e.g. to change RoPE scaling);
* without ``model``, the architecture comes from the checkpoint's ``config.json``. A partial
  ``model`` section (no ``block_types``), e.g. ``dict(grad_checkpointing=True)``, is applied on top;
* ``vocab_size='auto'`` takes the checkpoint's vocabulary size; ``tokenizer`` defaults to the
  checkpoint's own (``HFTokenizer``).
"""

from __future__ import annotations

import copy
import inspect
import os
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


def _pretrained_spec(spec) -> dict | None:
    if not spec:
        return None
    spec = dict(path=spec) if isinstance(spec, str) else dict(spec)
    unknown = set(spec) - {"path", "revision", "strict"}
    if "path" not in spec or unknown:
        raise KeyError(f"pretrained must be a path / Hub id or dict(path=..., revision=..., strict=...), got {spec}")
    spec.setdefault("revision", None)
    spec.setdefault("strict", True)
    return spec


def _resolve_pretrained(cfg: dict, spec: dict):
    """Fill ``model`` / ``tokenizer`` / ``vocab_size`` from the HF checkpoint's config. Returns that config."""
    from ..hf import config_from_hf, load_hf_config

    hf_config = load_hf_config(spec["path"], spec["revision"])
    model_cfg = cfg.get("model")
    if model_cfg is None or "block_types" not in model_cfg:  # absent or partial: the checkpoint's architecture
        cfg["model"] = {**config_from_hf(hf_config), **(model_cfg or {})}
    elif model_cfg.get("vocab_size", "auto") in (None, "auto"):
        cfg["model"]["vocab_size"] = hf_config.vocab_size
    cfg.setdefault("tokenizer", dict(type="HFTokenizer", name=spec["path"]))
    return hf_config


def _dataset_cfg(data_cfg: dict, objective: Objective) -> dict:
    data_cfg = dict(data_cfg)
    data_cfg.setdefault("type", objective.default_dataset)
    served = getattr(DATASETS.get(data_cfg["type"]), "objectives", None)
    if served is not None and objective.name not in served:
        raise ValueError(f"dataset type {data_cfg['type']!r} is for objective(s) {sorted(served)}, but the "
                         f"objective is {objective.name!r}; use {objective.default_dataset!r} or omit 'type'")
    return data_cfg


def _is_named_sets(val: dict) -> bool:
    return isinstance(val, dict) and bool(val) and "type" not in val and "path" not in val \
        and all(isinstance(v, dict) for v in val.values())


def _with_run_length(data_cfg: dict, train_cfg: dict) -> dict:
    """Datasets that take ``total_blocks`` (mixture schedules in fractions of training) get the run
    length in ``seq_len`` blocks: max_steps x batch x grad accumulation x world size."""
    params = inspect.signature(DATASETS.get(data_cfg["type"])).parameters
    if "total_blocks" not in params or data_cfg.get("total_blocks") is not None:
        return data_cfg
    steps = train_cfg.get("max_steps")
    if steps is None or steps <= 0:
        return data_cfg
    world = int(os.environ.get("WORLD_SIZE", 1))
    blocks = steps * train_cfg.get("per_device_train_batch_size", 8) * train_cfg.get("gradient_accumulation_steps", 1)
    return dict(data_cfg, total_blocks=blocks * world)


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
    pretrained = _pretrained_spec(cfg.get("pretrained"))
    if pretrained and cfg.get("init_from"):
        raise ValueError("set either init_from (a BrickFormer checkpoint) or pretrained (HF weights), not both")
    hf_config = None
    if cfg.get("init_from"):
        _resolve_init_from(cfg)
    elif pretrained:
        hf_config = _resolve_pretrained(cfg, pretrained)
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
    elif pretrained:
        from ..hf import load_hf_weights

        load_hf_weights(model, pretrained["path"], pretrained["revision"], pretrained["strict"], hf_config=hf_config)

    unknown = set(cfg["data"]) - {"train", "val"}
    if unknown:
        raise KeyError(f"data section accepts 'train' and 'val', got {sorted(unknown)}")
    train_cfg = _with_run_length(_dataset_cfg(cfg["data"]["train"], objective), cfg["train"])
    val = cfg["data"].get("val")
    if val is None:
        val_cfg = None
    elif _is_named_sets(val):  # val=dict(web=dict(...), math=dict(...)) -> eval_web_loss, eval_math_loss
        val_cfg = {name: _dataset_cfg(v, objective) for name, v in val.items()}
    else:
        val_cfg = _dataset_cfg(val, objective)
    train_ds = eval_ds = None
    if build_data:
        train_ds = DATASETS.build(train_cfg, tokenizer=tokenizer)
        if isinstance(val_cfg, dict) and "type" not in val_cfg:
            eval_ds = {name: DATASETS.build(v, tokenizer=tokenizer) for name, v in val_cfg.items()}
        elif val_cfg is not None:
            eval_ds = DATASETS.build(val_cfg, tokenizer=tokenizer)
        objective.source_names = getattr(train_ds, "source_names", None)
    runner = RUNNERS.build(cfg["train"])
    return Experiment(cfg, objective, model, tokenizer, train_ds, eval_ds, runner)


def run_experiment(cfg: dict):
    return build_experiment(cfg).run()
