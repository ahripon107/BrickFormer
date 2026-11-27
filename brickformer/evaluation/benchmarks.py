"""Zero-shot benchmarks scored by log-likelihood, as in lm-evaluation-harness.

Each example is a list of ``(context, continuation)`` pairs, one per answer choice, and the index of
the correct one. A choice's score is the summed log-probability of its continuation tokens given the
context:

* ``acc``: the highest-scoring choice is the gold one;
* ``acc_norm``: the same with scores divided by the answer's length in characters, which removes
  the bias toward short answers (the usual headline metric for HellaSwag, ARC-Challenge, PIQA,
  OpenBookQA);
* LAMBADA has a single continuation (the last word): ``acc`` is whether greedy decoding produces
  it exactly, ``ppl`` its perplexity.

Prompts, tokenization and metrics follow lm-evaluation-harness. On Qwen2.5-0.5B (full splits) the
results agree with lm-eval 0.4.13 to within 0.2 points (``prefix_eos=True`` puts EOS, the document
separator used in pretraining, before every context; that moves results by up to ~3 points).

| task          | dataset (split)                           | metrics          | primary  |
|---------------|-------------------------------------------|------------------|----------|
| hellaswag     | Rowan/hellaswag (validation)              | acc, acc_norm    | acc_norm |
| arc_easy      | allenai/ai2_arc ARC-Easy (test)           | acc, acc_norm    | acc      |
| arc_challenge | allenai/ai2_arc ARC-Challenge (test)      | acc, acc_norm    | acc_norm |
| piqa          | baber/piqa (validation)                   | acc, acc_norm    | acc_norm |
| winogrande    | allenai/winogrande winogrande_xl (val.)   | acc              | acc      |
| openbookqa    | allenai/openbookqa main (test)            | acc, acc_norm    | acc_norm |
| sciq          | allenai/sciq (test)                       | acc, acc_norm    | acc      |
| boolq         | google/boolq (validation)                 | acc              | acc      |
| mmlu          | cais/mmlu all (test; 5-shot from dev)     | acc              | acc      |
| lambada       | EleutherAI/lambada_openai (test)          | acc, ppl         | acc      |

Register your own task with ``@BENCHMARKS.register_module()`` on a :class:`Benchmark` subclass.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..registry import BENCHMARKS


@dataclass
class Example:
    requests: list[tuple[str, str]]  # (context, continuation) per choice
    gold: int


class Benchmark:
    """A dataset split turned into :class:`Example` s. Subclasses set the class attributes and
    implement :meth:`example`; any attribute can be overridden from the config, e.g.
    ``dict(type='hellaswag', path='/data/hellaswag', split='train')``."""

    path: str
    name: str | None = None
    split: str
    metrics: tuple[str, ...] = ("acc", "acc_norm")
    primary: str = "acc"
    greedy: bool = False  # single-continuation task scored by exact greedy match (LAMBADA)

    def __init__(self, path: str | None = None, name: str | None = None, split: str | None = None,
                 load_kwargs: dict | None = None):
        self.path = path or self.path
        self.name = name if name is not None else type(self).name
        self.split = split or self.split
        self.load_kwargs = load_kwargs or {}

    def example(self, row: dict) -> Example:
        raise NotImplementedError

    def load(self, max_samples: int | None = None, seed: int = 0) -> list[Example]:
        from ..data.datasets import load_split

        ds = load_split(self.path, self.name, self.split, load_kwargs=self.load_kwargs)
        if max_samples is not None and max_samples < len(ds):  # a fixed random subset, the same every eval
            ds = ds.shuffle(seed=seed).select(range(max_samples))
        return [self.example(row) for row in ds]


def _choices(context: str, options: list[str], gold: int) -> Example:
    return Example([(context, " " + o) for o in options], gold)


@BENCHMARKS.register_module(name="hellaswag")
class HellaSwag(Benchmark):
    path, split, primary = "Rowan/hellaswag", "validation", "acc_norm"

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip().replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    def example(self, row):
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        return _choices(self._clean(row["activity_label"] + ": " + ctx), [self._clean(e) for e in row["endings"]],
                        int(row["label"]))


class _ARC(Benchmark):
    path, split = "allenai/ai2_arc", "test"

    def example(self, row):
        labels = list(row["choices"]["label"])
        return _choices(f"Question: {row['question']}\nAnswer:", list(row["choices"]["text"]),
                        labels.index(row["answerKey"]))


@BENCHMARKS.register_module(name="arc_easy")
class ARCEasy(_ARC):
    name = "ARC-Easy"


@BENCHMARKS.register_module(name="arc_challenge")
class ARCChallenge(_ARC):
    name, primary = "ARC-Challenge", "acc_norm"


@BENCHMARKS.register_module(name="piqa")
class PIQA(Benchmark):
    path, split, primary = "baber/piqa", "validation", "acc_norm"

    def example(self, row):
        return _choices(f"Question: {row['goal']}\nAnswer:", [row["sol1"], row["sol2"]], int(row["label"]))


@BENCHMARKS.register_module(name="winogrande")
class WinoGrande(Benchmark):
    """Partial scoring: each option fills the blank in the context; the shared rest of the
    sentence is the continuation."""

    path, name, split, metrics = "allenai/winogrande", "winogrande_xl", "validation", ("acc",)

    def example(self, row):
        i = row["sentence"].index("_")
        rest = " " + row["sentence"][i + 1:].strip()
        return Example([(row["sentence"][:i] + row[f"option{k}"], rest) for k in (1, 2)], int(row["answer"]) - 1)


@BENCHMARKS.register_module(name="openbookqa")
class OpenBookQA(Benchmark):
    path, name, split, primary = "allenai/openbookqa", "main", "test", "acc_norm"

    def example(self, row):
        labels = list(row["choices"]["label"])
        return _choices(row["question_stem"], list(row["choices"]["text"]), labels.index(row["answerKey"].strip()))


@BENCHMARKS.register_module(name="sciq")
class SciQ(Benchmark):
    path, split = "allenai/sciq", "test"

    def example(self, row):
        ctx = f"{row['support'].lstrip()}\nQuestion: {row['question']}\nAnswer:"
        return _choices(ctx, [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]], 3)


@BENCHMARKS.register_module(name="boolq")
class BoolQ(Benchmark):
    path, split, metrics = "google/boolq", "validation", ("acc",)

    def example(self, row):
        return _choices(f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:", ["no", "yes"],
                        int(bool(row["answer"])))


@BENCHMARKS.register_module(name="mmlu")
class MMLU(Benchmark):
    """57 subjects, 4 choices, scored on the answer letter (" A" .. " D"). ``num_fewshot`` solved
    examples come first: the first ones of the subject's ``dev`` split, as in lm-eval and the
    original MMLU. ``acc`` is over all questions (each question counts once)."""

    path, name, split, metrics, num_fewshot = "cais/mmlu", "all", "test", ("acc",), 5

    def __init__(self, num_fewshot: int | None = None, fewshot_split: str = "dev", **kwargs):
        super().__init__(**kwargs)
        self.num_fewshot = type(self).num_fewshot if num_fewshot is None else num_fewshot
        self.fewshot_split = fewshot_split
        self._shots: dict[str, str] = {}

    @staticmethod
    def _question(row: dict) -> str:
        options = "".join(f"\n{letter}. {c}" for letter, c in zip("ABCD", row["choices"]))
        return f"{row['question'].strip()}{options}\nAnswer:"

    def load(self, max_samples=None, seed=0):
        from ..data.datasets import load_split

        if self.num_fewshot:
            shots: dict[str, list[str]] = {}
            for row in load_split(self.path, self.name, self.fewshot_split, load_kwargs=self.load_kwargs):
                if len(shots.setdefault(row["subject"], [])) < self.num_fewshot:
                    shots[row["subject"]].append(f"{self._question(row)} {'ABCD'[row['answer']]}\n\n")
            self._shots = {s: "".join(v) for s, v in shots.items()}
        return super().load(max_samples, seed)

    def example(self, row):
        subject = row["subject"]
        header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        return _choices(header + self._shots.get(subject, "") + self._question(row), list("ABCD"), int(row["answer"]))


@BENCHMARKS.register_module(name="lambada")
class LAMBADA(Benchmark):
    path, name, split, metrics, greedy = "EleutherAI/lambada_openai", "default", "test", ("acc", "ppl"), True

    def example(self, row):
        ctx, word = row["text"].rsplit(" ", 1)
        return Example([(ctx, " " + word)], 0)


# ------------------------------------------------------------------------------------------ scoring
def _encode(tokenizer, context: str, continuation: str, max_len: int,
            prefix_eos: bool = False) -> tuple[list[int], int]:
    """``context + continuation`` ids (cut from the left to ``max_len``) and the number of
    continuation tokens, tokenized as lm-eval does: trailing whitespace moves from the context to
    the continuation, the pair is encoded as one string, and the continuation is what follows the
    context's own tokens. If a token merges across the boundary (the joint encoding does not start
    with the context's tokens), the continuation is encoded on its own instead. EOS goes in front of
    an empty context (something must predict the first token), or of every context with
    ``prefix_eos=True``."""
    stripped = context.rstrip()
    continuation = context[len(stripped):] + continuation
    ctx = tokenizer.encode(stripped)
    whole = tokenizer.encode(stripped + continuation)
    cont = whole[len(ctx):] if whole[:len(ctx)] == ctx else []
    if not cont:
        cont = tokenizer.encode(continuation)
    if not cont:
        raise ValueError(f"empty continuation for context {context[-50:]!r}")
    if prefix_eos or not ctx:
        ctx = [tokenizer.eos_token_id] + ctx
    ids = (ctx + cont)[-(max_len + 1):]  # +1: the last token is only a target
    return ids, min(len(cont), len(ids) - 1)


@torch.no_grad()
def score_requests(model, tokenizer, requests: list[tuple[str, str]], max_len: int = 1024,
                   batch_tokens: int = 4096, autocast_dtype: torch.dtype | None = None,
                   prefix_eos: bool = False) -> list[tuple[float, bool]]:
    """``(sum log p(continuation | context), whether greedy decoding yields the continuation)`` per
    request. Batches hold about ``batch_tokens`` tokens (memory: about ``batch_tokens x vocab``
    logits), sorted by length so padding is small; right padding leaves causal logits unchanged.
    Requests with identical inputs (one context, single-token choices such as MMLU's letters)
    share one forward pass."""
    device = next(model.parameters()).device
    encoded = [_encode(tokenizer, c, x, max_len, prefix_eos) for c, x in requests]
    rows: dict[tuple[int, ...], list[int]] = {}  # model input -> the requests that score on it
    for i, (ids, _) in enumerate(encoded):
        rows.setdefault(tuple(ids[:-1]), []).append(i)
    inputs = sorted(rows, key=len, reverse=True)
    results: list = [None] * len(encoded)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    start = 0
    while start < len(inputs):
        width = len(inputs[start])  # longest in the batch (sorted)
        size = max(1, batch_tokens // max(width, 1))
        batch = inputs[start:start + size]
        start += size
        x = torch.full((len(batch), width), pad, dtype=torch.long)
        for r, inp in enumerate(batch):
            x[r, :len(inp)] = torch.tensor(inp)
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = model(x.to(device)).logits
        for r, inp in enumerate(batch):
            for i in rows[inp]:
                ids, n = encoded[i]
                pos = torch.arange(len(ids) - 1 - n, len(ids) - 1, device=device)
                target = torch.tensor(ids[-n:], device=device)
                logp = F.log_softmax(logits[r, pos].float(), dim=-1)
                results[i] = (logp.gather(-1, target[:, None]).sum().item(), bool((logp.argmax(-1) == target).all()))
    return results


def evaluate_benchmark(model, tokenizer, task: Benchmark, examples: list[Example], **score_kwargs) -> dict[str, float]:
    requests = [req for ex in examples for req in ex.requests]
    scores = iter(score_requests(model, tokenizer, requests, **score_kwargs))
    hits = {m: 0 for m in task.metrics if m != "ppl"}
    nll = 0.0
    for ex in examples:
        res = [next(scores) for _ in ex.requests]
        if task.greedy:
            hits["acc"] += res[ex.gold][1]
            nll -= res[ex.gold][0]
            continue
        logps = [lp for lp, _ in res]
        hits["acc"] += max(range(len(logps)), key=logps.__getitem__) == ex.gold
        if "acc_norm" in hits:
            norm = [lp / max(len(cont.lstrip(" ")), 1) for lp, (_, cont) in zip(logps, ex.requests)]
            hits["acc_norm"] += max(range(len(norm)), key=norm.__getitem__) == ex.gold
    out = {m: h / len(examples) for m, h in hits.items()}
    if "ppl" in task.metrics:
        out["ppl"] = math.exp(nll / len(examples))
    return out


class BenchmarkSuite:
    """The ``benchmarks`` config section::

        benchmarks = dict(
            tasks=['hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'winogrande', 'lambada'],
            max_samples=500,      # per task: a fixed random subset (seeded); None = the full split
            every=None,           # optimizer steps between runs; default train.eval_steps; 0 = at the end only
            batch_tokens=4096,    # tokens per forward pass (logits memory ~ batch_tokens x vocab x 2 bytes)
            max_len=1024,         # context is cut from the left to fit
            seed=0,
            prefix_eos=False,     # True: EOS before every context (default: as lm-eval)
        )

    A task is a registered name or ``dict(type=name, path=..., name=..., split=...)``. Results are
    logged as ``bench/<task>_<metric>`` plus ``bench/avg`` (the mean of each task's primary metric).
    """

    def __init__(self, tasks: list, max_samples: int | None = 500, every: int | None = None,
                 batch_tokens: int = 4096, max_len: int = 1024, seed: int = 0, prefix_eos: bool = False):
        if not tasks:
            raise ValueError("benchmarks.tasks is empty")
        self.tasks: dict[str, Benchmark] = {}
        for t in tasks:
            cfg = dict(type=t) if isinstance(t, str) else dict(t)
            label = cfg.pop("label", cfg["type"])
            if label in self.tasks:
                raise ValueError(f"benchmark {label!r} is listed twice; give one a different label=")
            self.tasks[label] = BENCHMARKS.build(cfg)
        self.max_samples, self.every, self.seed = max_samples, every, seed
        self.batch_tokens, self.max_len, self.prefix_eos = batch_tokens, max_len, prefix_eos
        self._examples: dict[str, list[Example]] | None = None

    def load(self) -> None:
        if self._examples is None:
            self._examples = {name: t.load(self.max_samples, self.seed) for name, t in self.tasks.items()}

    def evaluate(self, model, tokenizer, autocast_dtype: torch.dtype | None = None) -> dict[str, float]:
        """Runs every task on ``model`` (put in eval mode, restored afterwards)."""
        self.load()
        was_training = model.training
        model.eval()
        out, primary = {}, []
        t0 = time.time()
        try:
            for name, task in self.tasks.items():
                res = evaluate_benchmark(model, tokenizer, task, self._examples[name], max_len=self.max_len,
                                         batch_tokens=self.batch_tokens, autocast_dtype=autocast_dtype,
                                         prefix_eos=self.prefix_eos)
                out.update({f"bench/{name}_{m}": round(v, 6) for m, v in res.items()})
                primary.append(res[task.primary])
        finally:
            model.train(was_training)
        out["bench/avg"] = round(sum(primary) / len(primary), 6)
        out["bench/runtime"] = round(time.time() - t0, 2)
        return out
