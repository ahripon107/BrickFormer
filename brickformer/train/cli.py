"""``brickformer-train CONFIG [--cfg-options k=v ...] [--resume [PATH]] [--dry-run]``"""

from __future__ import annotations

import argparse
import json

from ..config import apply_overrides, load_config
from ..utils import summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="brickformer-train", description="Train a BrickFormer experiment config.")
    ap.add_argument("config", help="experiment config (.py / .yaml / .json) with model, tokenizer, data, train")
    ap.add_argument("--cfg-options", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. train.learning_rate=1e-3 model.dim=512")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume from the latest checkpoint in output_dir, or from the given path")
    ap.add_argument("--dry-run", action="store_true",
                    help="build tokenizer, model and runner, print the resolved config and summary, then exit")
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.cfg_options)
    if args.resume is not None:
        cfg["train"]["resume_from_checkpoint"] = args.resume

    from .experiment import build_experiment

    exp = build_experiment(cfg, build_data=not args.dry_run)
    seq_len = cfg["data"]["train"].get("seq_len") or cfg["data"]["train"].get("max_len", 2048)
    data_type = exp.config["data"]["train"].get("type", exp.objective.default_dataset)
    print(f"objective: {exp.objective.name} | train data: {data_type}")
    print(summary(exp.model, seq_len))
    if args.dry_run:
        print(json.dumps(exp.config, indent=2, default=str))
        return
    exp.run()


if __name__ == "__main__":
    main()
