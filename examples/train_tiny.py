"""Minimal single-GPU training loop: a character-level LM on tiny-shakespeare.

    python examples/train_tiny.py                          # built-in GDN/attention hybrid
    python examples/train_tiny.py --pattern "MM*"          # Mamba2 + attention
    python examples/train_tiny.py --config configs/llama_like.py --dim 256 --layers 4

``--config`` takes any BrickFormer config; ``vocab_size`` is overridden by the dataset's
character vocabulary, and ``--dim`` / ``--layers`` optionally shrink it.
"""

from __future__ import annotations

import argparse
import math
import time
import urllib.request
from pathlib import Path

import torch

import brickformer as bf

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def tiny_config(dim: int, pattern: str, layers: int) -> dict:
    heads = max(dim // 64, 1)
    ffn = dict(type="SwiGLU", hidden=int(dim * 8 / 3) // 64 * 64)
    return dict(
        type="CausalLM", vocab_size=0, dim=dim, pattern=pattern, num_layers=layers,
        block_types=dict(
            attn=dict(mixer=dict(type="GQAttention", heads=heads, kv_heads=max(heads // 2, 1)), ffn=ffn),
            swa=dict(mixer=dict(type="SlidingWindowAttention", heads=heads, window=64), ffn=ffn),
            gdn=dict(mixer=dict(type="GatedDeltaNet", heads=heads, head_dim=64), ffn=ffn),
            mamba=dict(mixer=dict(type="Mamba2", d_state=32, headdim=32), ffn=ffn),
            mla=dict(mixer=dict(type="MLA", heads=heads, kv_lora_rank=dim // 2, nope_head_dim=32,
                                rope_head_dim=32, v_head_dim=64), ffn=ffn),
        ),
        tie_embeddings=True,
    )


def load_text(path: str | None) -> str:
    if path:
        return Path(path).read_text()
    cache = Path.home() / ".cache" / "brickformer" / "tinyshakespeare.txt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading tiny-shakespeare to {cache}")
        urllib.request.urlretrieve(DATA_URL, cache)
    return cache.read_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="BrickFormer config file (.py/.yaml/.json)")
    ap.add_argument("--pattern", default="GGG*", help="pattern for the built-in config")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--data", help="path to a text file (default: tiny-shakespeare)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", default="runs/tiny")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    text = load_text(args.data)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]

    if args.config:
        cfg = bf.load_config(args.config)
        cfg = cfg.get("model", cfg)
        if args.layers and "num_layers" in cfg:
            cfg["num_layers"] = args.layers
        if args.dim:
            cfg["dim"] = args.dim
    else:
        cfg = tiny_config(args.dim, args.pattern, args.layers)
    cfg["vocab_size"] = len(chars)
    model = bf.build_model(cfg).to(device)
    print(bf.summary(model, args.seq_len))

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(s / args.steps, 1.0))))
    fwd = torch.compile(model) if args.compile else model
    amp = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")

    def batch(src):
        ix = torch.randint(len(src) - args.seq_len - 1, (args.batch_size,))
        return torch.stack([src[i:i + args.seq_len] for i in ix]).to(device, non_blocking=True)

    @torch.no_grad()
    def evaluate(n=20):
        model.eval()
        losses = []
        for _ in range(n):
            x = batch(val)
            with amp:
                losses.append(fwd(x, labels=x).loss.item())
        model.train()
        return sum(losses) / n

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x = batch(train)
        with amp:
            out = fwd(x, labels=x)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 20 == 0:
            tok_s = 20 * args.batch_size * args.seq_len / (time.time() - t0)
            aux = f" aux {out.aux_loss.item():.4f}" if out.aux_loss is not None else ""
            lr = sched.get_last_lr()[0]
            print(f"step {step:5d} | loss {out.loss.item():.4f}{aux} | lr {lr:.2e} | {tok_s:,.0f} tok/s")
            t0 = time.time()
        if step % args.eval_every == 0 or step == args.steps:
            print(f"step {step:5d} | val loss {evaluate():.4f}")

    model.eval()
    prompt = torch.tensor([[stoi[c] for c in "ROMEO:\n"]], device=device)
    with amp:
        sample = model.generate(prompt, 300, temperature=0.8, top_p=0.95)
    print("-" * 60 + "\n" + "".join(chars[i] for i in sample[0].tolist()))
    bf.save_pretrained(model, args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
