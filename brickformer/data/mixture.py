"""Pretraining on a mixture of sources whose weights follow a schedule over training.

::

    data = dict(train=dict(type='MixturePretrainDataset', seq_len=1024,
        sources=dict(
            web=dict(path='HuggingFaceFW/fineweb-edu', name='sample-10BT', streaming=True),
            math=dict(path='open-web-math/open-web-math', streaming=True),
            code=dict(path='bigcode/the-stack-smol', text_field='content', streaming=True)),
        schedule=[                                   # step phases: weights hold until `until`
            dict(until=0.8, weights=dict(web=0.85, math=0.10, code=0.05)),
            dict(until=1.0, weights=dict(web=0.40, math=0.35, code=0.25)),
        ]))

Schedules:

* ``weights=dict(...)``: one fixed mixture;
* step phases ``dict(until=..., weights=...)``: phase *i* runs up to ``until`` (a fraction of
  training) or ``until_tokens`` (absolute tokens); the last phase's weights continue past its end;
* linear keypoints ``dict(at=..., weights=...)`` / ``at_tokens``: weights are interpolated linearly
  between keypoints and held constant before the first and after the last.

Weights are relative (normalised per position) and a source left out of a phase has weight 0.
Fractions of training need the run length, which the experiment fills in as ``total_blocks``
(``max_steps x batch x grad accumulation x world size``).

Each source is packed separately, so every ``seq_len`` block comes from exactly one source and the
mixture holds exactly in tokens. Which source each block comes from is drawn from a generator seeded
with ``seed``, so the stream is reproducible (resuming replays it) and identical with any number of
dataloader workers. A source that runs out restarts (``on_exhausted='repeat'``, reshuffling
streaming sources) or ends training data (``'stop'``).

Each block carries its ``source`` index; the collator passes it on and the trainer logs the share of
each source per logging interval as ``mix/<name>``.
"""

from __future__ import annotations

import bisect
import itertools

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..registry import DATASETS
from .datasets import _objectives, load_split, load_stream

_SOURCE_KEYS = {"path", "name", "split", "data_files", "text_field", "streaming", "shuffle_buffer",
                "max_samples", "load_kwargs"}


class MixtureSchedule:
    """Source weights as a function of the block index."""

    def __init__(self, names: list[str], weights: dict | None = None, schedule: list[dict] | None = None,
                 seq_len: int = 1, total_blocks: int | None = None):
        if (weights is None) == (schedule is None):
            raise ValueError("give exactly one of weights=dict(...) or schedule=[...]")
        self.names = list(names)
        if weights is not None:
            schedule = [dict(at_tokens=0, weights=weights)]  # constant: needs no run length
        if not schedule:
            raise ValueError("schedule is empty")
        kinds = {("until" in p or "until_tokens" in p, "at" in p or "at_tokens" in p) for p in schedule}
        if kinds == {(True, False)}:
            self.linear = False
        elif kinds == {(False, True)}:
            self.linear = True
        else:
            raise ValueError("schedule entries must all be step phases (until / until_tokens) or all linear "
                             "keypoints (at / at_tokens), each with exactly one position key")
        self.points, self.vectors = [], []
        for p in schedule:
            self.points.append(self._position(p, seq_len, total_blocks))
            self.vectors.append(self._vector(p.get("weights")))
        if any(b <= a for a, b in zip(self.points, self.points[1:])):
            raise ValueError(f"schedule positions must increase, got blocks {self.points}")

    def _position(self, phase: dict, seq_len: int, total_blocks: int | None) -> float:
        keys = [k for k in ("until", "until_tokens", "at", "at_tokens") if k in phase]
        if len(keys) != 1 or set(phase) - {keys[0], "weights"}:
            raise ValueError(f"each schedule entry needs one position key and weights, got {phase}")
        value = phase[keys[0]]
        if keys[0].endswith("_tokens"):
            return float(value) / seq_len
        if total_blocks is None:
            raise ValueError(f"schedule position {keys[0]}={value} is a fraction of training, which needs the "
                             "run length: set train.max_steps, or use until_tokens / at_tokens")
        if not 0 <= value <= 1:
            raise ValueError(f"{keys[0]}={value} must be a fraction in [0, 1]; use {keys[0]}_tokens for tokens")
        return float(value) * total_blocks

    def _vector(self, weights) -> np.ndarray:
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"weights must be a non-empty dict of source -> weight, got {weights!r}")
        unknown = set(weights) - set(self.names)
        if unknown:
            raise KeyError(f"weights name unknown sources {sorted(unknown)}; sources are {self.names}")
        v = np.array([float(weights.get(n, 0.0)) for n in self.names])
        if (v < 0).any() or v.sum() <= 0:
            raise ValueError(f"weights must be >= 0 with a positive sum, got {weights}")
        return v / v.sum()

    def weights_at(self, block: int) -> np.ndarray:
        if not self.linear:  # phase i covers (points[i-1], points[i]]; past the end: the last phase
            i = min(bisect.bisect_left(self.points, block + 1), len(self.points) - 1)
            return self.vectors[i]
        if block <= self.points[0]:
            return self.vectors[0]
        if block >= self.points[-1]:
            return self.vectors[-1]
        i = bisect.bisect_right(self.points, block)
        a, b = self.points[i - 1], self.points[i]
        t = (block - a) / (b - a)
        return (1 - t) * self.vectors[i - 1] + t * self.vectors[i]

    def as_dict(self, block: int) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.weights_at(block))))


class _SourceStream:
    """Packed ``seq_len`` blocks from one source, restarting (with a new shuffle) when exhausted."""

    def __init__(self, ds, tokenizer, text_field: str, seq_len: int, max_samples, repeat: bool,
                 worker: int, num_workers: int, seed: int):
        self.ds, self.tokenizer, self.text_field, self.seq_len = ds, tokenizer, text_field, seq_len
        self.seed = seed
        self.max_samples, self.repeat = max_samples, repeat
        self.worker, self.num_workers = worker, num_workers
        self.epoch = 0
        self._blocks = self._iter()

    def _examples(self):
        ds = self.ds
        if hasattr(ds, "set_epoch"):  # streaming: a new shuffle-buffer order per pass
            ds.set_epoch(self.epoch)
        elif self.epoch:  # map-style: reshuffle on every repeat
            ds = ds.shuffle(seed=self.seed + self.epoch)
        examples = iter(ds)
        if self.max_samples is not None:
            examples = itertools.islice(examples, self.max_samples)
        if self.num_workers > 1:  # disjoint documents per worker
            examples = itertools.islice(examples, self.worker, None, self.num_workers)
        return examples

    def _iter(self):
        eos, buf = self.tokenizer.eos_token_id, []
        while True:
            produced = False
            for ex in self._examples():
                buf.extend(self.tokenizer.encode(ex[self.text_field]) + [eos])
                start = 0
                while len(buf) - start >= self.seq_len:
                    yield buf[start:start + self.seq_len]
                    produced = True
                    start += self.seq_len
                buf = buf[start:]
            if not self.repeat or not produced:
                return
            self.epoch += 1

    def next(self):
        return next(self._blocks, None)


@_objectives("pretrain")
class MixturePretrainDataset(IterableDataset):
    """See the module docstring. ``sources``: name -> ``dict(path, name, split, data_files, text_field,
    streaming, shuffle_buffer, max_samples, load_kwargs)``."""

    def __init__(self, sources: dict, tokenizer, seq_len: int, weights: dict | None = None,
                 schedule: list[dict] | None = None, total_blocks: int | None = None, seed: int = 42,
                 on_exhausted: str = "repeat", text_field: str = "text"):
        if not sources:
            raise ValueError("sources is empty")
        if on_exhausted not in ("repeat", "stop"):
            raise ValueError(f"on_exhausted must be 'repeat' or 'stop', got {on_exhausted!r}")
        for name, src in sources.items():
            unknown = set(src) - _SOURCE_KEYS
            if "path" not in src or unknown:
                raise KeyError(f"source {name!r} needs a path and accepts {sorted(_SOURCE_KEYS)}; "
                               f"unknown keys: {sorted(unknown)}")
        self.source_names = list(sources)
        self.sources, self.tokenizer, self.seq_len = dict(sources), tokenizer, seq_len
        self.schedule = MixtureSchedule(self.source_names, weights, schedule, seq_len, total_blocks)
        self.seed, self.on_exhausted, self.text_field = seed, on_exhausted, text_field
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _load(self, src: dict):
        args = (src["path"], src.get("name"), src.get("split", "train"), src.get("data_files"))
        if src.get("streaming", False):
            return load_stream(*args, src.get("shuffle_buffer"), self.seed, src.get("load_kwargs"))
        return load_split(*args, False, src.get("max_samples"), src.get("load_kwargs")).shuffle(seed=self.seed)

    def __iter__(self):
        info = get_worker_info()
        worker, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        streams = [
            _SourceStream(self._load(src), self.tokenizer, src.get("text_field", self.text_field), self.seq_len,
                          src.get("max_samples") if src.get("streaming") else None,
                          self.on_exhausted == "repeat", worker, num_workers, self.seed)
            for src in self.sources.values()
        ]
        rng = np.random.default_rng([self.seed, self.epoch])
        for block in itertools.count():
            choice = int(rng.choice(len(streams), p=self.schedule.weights_at(block)))
            if block % num_workers != worker:
                continue  # every worker draws the same sequence and yields its own share, in order
            ids = streams[choice].next()
            if ids is None:  # exhausted ('stop', or a source that yields nothing)
                if self.on_exhausted == "stop":
                    return
                raise ValueError(f"source {self.source_names[choice]!r} produced no {self.seq_len}-token blocks")
            yield {"input_ids": torch.tensor(ids), "source": choice}


DATASETS.register_module(name="MixturePretrainDataset")(MixturePretrainDataset)
