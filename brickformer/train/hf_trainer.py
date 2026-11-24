"""HF ``transformers.Trainer`` backend.

``BrickTrainer`` delegates the loss to the experiment's objective (pretrain / sft / dpo / grpo) and
overrides what the Trainer only knows for HF ``PreTrainedModel`` checkpoints:

* saving: ``bf.save_pretrained`` format (``config.json`` + ``model.pt``), which handles tied
  embeddings and loads back with ``bf.load_pretrained(checkpoint_dir)``; the experiment config
  and tokenizer are saved next to it;
* loading (resume and ``load_best_model_at_end``) from that format;
* evaluation: always through the objective's loss (``prediction_loss_only``);
* logging: objectives ``record()`` extra metrics (MoE ``aux_loss``, DPO reward margins, GRPO
  rewards...), which are averaged and logged with the train loss, or with ``eval_`` prefixes.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import torch

from ..config import dump_config, save_pretrained
from ..registry import RUNNERS

try:
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
except ImportError:  # pragma: no cover - exercised only without transformers
    Trainer = None


def _require_transformers():
    if Trainer is None:
        raise ImportError("the HFTrainer runner requires `pip install transformers accelerate` "
                          "(or `pip install brickformer[train]`)")


if Trainer is not None:

    class BrickTrainer(Trainer):
        def __init__(self, *args, objective, tokenizer, ref_model=None, experiment_config: dict | None = None,
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.objective = objective
            self.bf_tokenizer = tokenizer
            self.experiment_config = experiment_config
            self.ref_model = ref_model.to(self.args.device) if ref_model is not None else None
            self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # ------------------------------------------------------------------ loss and metrics
        def record(self, name: str, value) -> None:
            """Called by objectives; values are averaged into the next train / eval log."""
            mode = "train" if self.model.training else "eval"
            value = value.detach().float() if torch.is_tensor(value) else torch.tensor(float(value))
            self._metrics[mode][name].append(value.to(self.args.device))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = self.objective.compute_loss(self, model, inputs)
            return (loss, None) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad(), self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

        def log(self, logs: dict, start_time=None) -> None:
            for mode, key, prefix in (("train", "loss", ""), ("eval", "eval_loss", "eval_")):
                if key in logs and self._metrics[mode]:
                    for name, values in self._metrics[mode].items():
                        logs[prefix + name] = round(torch.stack(values).mean().item(), 6)
                    self._metrics[mode].clear()
            super().log(logs, start_time)

        # ------------------------------------------------------------------ checkpoints
        def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
            output_dir = Path(output_dir or self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            if state_dict is None:
                save_pretrained(model, output_dir)
            else:
                dump_config(model.config, output_dir / "config.json")
                torch.save(state_dict, output_dir / "model.pt")
            if self.experiment_config is not None:
                dump_config(self.experiment_config, output_dir / "experiment.json")
            if self.bf_tokenizer is not None and hasattr(self.bf_tokenizer, "save_pretrained"):
                self.bf_tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        def _load_weights(self, directory: str, model=None) -> None:
            path = Path(directory) / "model.pt"
            if not path.is_file():
                raise ValueError(f"no BrickFormer checkpoint (model.pt) in {directory}")
            model = model if model is not None else self.model
            model.load_state_dict(torch.load(path, map_location="cpu"))

        def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
            self._load_weights(resume_from_checkpoint, model)

        def _load_best_model(self) -> None:
            self._load_weights(self.state.best_model_checkpoint)


@RUNNERS.register_module()
class HFTrainer:
    """Config section ``train = dict(type='HFTrainer', output_dir=..., **TrainingArguments)``.

    Any ``transformers.TrainingArguments`` field can be given. Defaults differ from HF where it
    matters for LM training: ``prediction_loss_only=True`` (never gather vocab-sized logits),
    ``report_to='none'``, ``remove_unused_columns=False``, and evaluation every ``eval_steps``
    (default: ``logging_steps``) when a validation set is configured.

    ``resume_from_checkpoint``: ``None``, ``True`` / ``'auto'`` (latest checkpoint in
    ``output_dir`` if any), or a checkpoint path.
    """

    def __init__(self, output_dir: str, resume_from_checkpoint: str | bool | None = None, **training_args):
        _require_transformers()
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.training_args = training_args

    def make_args(self, has_eval: bool) -> TrainingArguments:
        args = dict(output_dir=self.output_dir, prediction_loss_only=True, report_to="none",
                    remove_unused_columns=False, logging_steps=50)
        args.update(self.training_args)
        fields = getattr(TrainingArguments, "__dataclass_fields__", {})
        if "warmup_ratio" in args and "warmup_ratio" not in fields:
            # transformers 5 folded warmup_ratio into warmup_steps (a float in [0, 1) is a ratio)
            ratio = args.pop("warmup_ratio")
            if ratio and not args.get("warmup_steps"):
                args["warmup_steps"] = float(ratio)
        if has_eval:
            args.setdefault("eval_strategy", "steps")
            args.setdefault("eval_steps", args["logging_steps"])
        return TrainingArguments(**args)

    def _resume_path(self) -> str | None:
        r = self.resume_from_checkpoint
        if r in (True, "auto"):
            return get_last_checkpoint(self.output_dir) if os.path.isdir(self.output_dir) else None
        return r or None

    def run(self, model, tokenizer, train_dataset, eval_dataset=None, objective=None,
            experiment_config: dict | None = None):
        if objective is None:
            from .objectives import build_objective

            objective = build_objective("pretrain")
        # the reference is the starting policy, so build it before any checkpoint is resumed
        ref_model = objective.build_ref_model(model) if objective.needs_ref_model else None
        trainer = BrickTrainer(
            model=model, args=self.make_args(eval_dataset is not None), train_dataset=train_dataset,
            eval_dataset=eval_dataset, data_collator=objective.collator(tokenizer), objective=objective,
            tokenizer=tokenizer, ref_model=ref_model, experiment_config=experiment_config,
        )
        trainer.train(resume_from_checkpoint=self._resume_path())
        trainer.save_model(self.output_dir)
        already_evaluated = any("eval_loss" in h and h.get("step") == trainer.state.global_step
                                for h in trainer.state.log_history)
        if eval_dataset is not None and not already_evaluated:
            trainer.evaluate()
        return trainer
