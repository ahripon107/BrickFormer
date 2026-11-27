"""``brickformer-eval (CHECKPOINT | --hf MODEL) [--tasks a,b,...] [--max-samples N] ...``

Scores a BrickFormer checkpoint (a training ``output_dir`` or ``checkpoint-*`` directory) or an HF
model (Llama / Mistral / Qwen2 / Qwen3, imported with ``from_hf``) on zero-shot benchmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

DEFAULT_TASKS = "hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada"


def _load_checkpoint(directory: Path):
    from ..config import load_config, load_pretrained
    from ..registry import TOKENIZERS

    exp = directory / "experiment.json"
    if not exp.is_file():
        raise FileNotFoundError(f"{directory} has no experiment.json to take the tokenizer from")
    return load_pretrained(directory), TOKENIZERS.build(load_config(exp)["tokenizer"])


def main(argv: list[str] | None = None) -> None:
    from .. import data as _data  # noqa: F401  (registers tokenizers)
    from .benchmarks import BenchmarkSuite

    ap = argparse.ArgumentParser(prog="brickformer-eval", description=__doc__.split("\n\n")[1])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("checkpoint", nargs="?", help="BrickFormer checkpoint directory")
    src.add_argument("--hf", metavar="MODEL", help="HF Hub id or local HF directory instead of a checkpoint")
    ap.add_argument("--tasks", default=DEFAULT_TASKS, help=f"comma-separated (default: {DEFAULT_TASKS})")
    ap.add_argument("--max-samples", type=int, default=None, help="per task (default: the full split)")
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="selects the --max-samples subset")
    ap.add_argument("--prefix-eos", action="store_true", help="put EOS before every context (default: as lm-eval)")
    ap.add_argument("--output", help="also write the results to this JSON file")
    args = ap.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    if args.hf:
        from ..data import HFTokenizer
        from ..hf import from_hf

        model, tokenizer = from_hf(args.hf), HFTokenizer(args.hf)
    else:
        model, tokenizer = _load_checkpoint(Path(args.checkpoint))
    model = model.to(args.device, dtype)

    suite = BenchmarkSuite(tasks=args.tasks.split(","), max_samples=args.max_samples, batch_tokens=args.batch_tokens,
                           max_len=args.max_len, seed=args.seed, prefix_eos=args.prefix_eos)
    results = suite.evaluate(model, tokenizer)
    width = max(map(len, results))
    for k, v in results.items():
        print(f"{k:<{width}}  {v:.4f}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
