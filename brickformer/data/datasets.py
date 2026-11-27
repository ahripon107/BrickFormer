"""Datasets for each training objective, built on Hugging Face ``datasets``.

| objective | dataset type        | produces                                   |
|-----------|---------------------|--------------------------------------------|
| pretrain  | ``PretrainDataset`` | packed ``input_ids`` blocks of ``seq_len`` |
| sft       | ``SFTDataset``      | ``input_ids`` + ``labels`` (prompt masked) |
| dpo       | ``PreferenceDataset`` | ``prompt_ids``, ``chosen_ids``, ``rejected_ids`` |
| grpo      | ``PromptDataset``   | ``prompt_ids`` + the raw columns rewards need |

All of them load Hub datasets (``path='openai/gsm8k'``) or local files (``path='json'`` /
``'parquet'`` / ``'text'`` with ``data_files=...``). ``prompt_template`` (e.g.
``'Question: {question}\\nAnswer:'``) builds the prompt from any columns of a row.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..registry import DATASETS
from .chat import encode_prompt, is_messages, split_prompt_response, tokenize_conversation


def _load_dataset(**kwargs):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("BrickFormer datasets require `pip install datasets` "
                          "(or `pip install brickformer[train]`)") from e
    return load_dataset(**kwargs)


def load_split(path: str, name: str | None = None, split: str = "train", data_files=None, streaming: bool = False,
               max_samples: int | None = None, load_kwargs: dict | None = None):
    kwargs = dict(path=path, name=name, split=split, data_files=data_files, streaming=streaming, **(load_kwargs or {}))
    ds = _load_dataset(**{k: v for k, v in kwargs.items() if v is not None})
    if max_samples is not None and not streaming:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


class ShuffledStream:
    """A streaming split shuffled by file order and a document buffer, reshuffled per epoch.

    Used instead of ``datasets.IterableDataset.shuffle``: on a stream of many parquet files that
    shuffles row groups across every file and held 5-6 GB of RAM for 13 FineWeb-Edu shards before the
    first example (datasets 5.2), versus 0.7 GB streamed in order. Here the ``data_files`` list is
    shuffled (when it is a list; a single file, a pattern or a split dict keeps its order), streamed
    in that order, and documents pass through a ``buffer_size`` shuffle buffer.
    """

    def __init__(self, path: str, name: str | None = None, split: str = "train", data_files=None,
                 buffer_size: int = 10_000, seed: int = 42, load_kwargs: dict | None = None):
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        self.path, self.name, self.split, self.load_kwargs = path, name, split, load_kwargs
        self.data_files = [data_files] if isinstance(data_files, str) else data_files
        self.buffer_size, self.seed, self.epoch = buffer_size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(f"{self.seed}-{self.epoch}")
        files = self.data_files
        if isinstance(files, (list, tuple)):
            files = list(files)
            rng.shuffle(files)
        ds = load_split(self.path, self.name, self.split, files, True, None, self.load_kwargs)
        buf: list = []
        for ex in ds:
            if len(buf) < self.buffer_size:
                buf.append(ex)
                continue
            i = rng.randrange(self.buffer_size)
            yield buf[i]
            buf[i] = ex
        rng.shuffle(buf)
        yield from buf


def load_stream(path: str, name: str | None = None, split: str = "train", data_files=None,
                shuffle_buffer: int | None = None, seed: int = 42, load_kwargs: dict | None = None):
    """A streaming split, shuffled with :class:`ShuffledStream` when ``shuffle_buffer`` is set."""
    if shuffle_buffer:
        return ShuffledStream(path, name, split, data_files, shuffle_buffer, seed, load_kwargs)
    return load_split(path, name, split, data_files, True, None, load_kwargs)


def _objectives(*names: str):
    """Marks which objectives a dataset type serves (checked when an experiment is built)."""
    def wrap(fn):
        fn.objectives = frozenset(names)
        return fn
    return wrap


def _prompt(row: dict, prompt_field: str, prompt_template: str | None) -> Any:
    return prompt_template.format(**row) if prompt_template else row[prompt_field]


def _truncate(prompt_ids: list[int], response_ids: list[int], max_len: int | None, max_prompt_len: int | None):
    if max_prompt_len is not None and len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]  # keep the end of the prompt, next to the response
    if max_len is not None:
        response_ids = response_ids[:max(max_len - len(prompt_ids), 0)]
    return prompt_ids, response_ids


# ------------------------------------------------------------------------------------------- pretrain
class PackedStream(IterableDataset):
    """Tokenizes and packs an iterable of examples on the fly (used for ``streaming=True``)."""

    def __init__(self, source, tokenizer, text_field: str, seq_len: int, packing: bool = True,
                 max_samples: int | None = None):
        self.source, self.tokenizer, self.text_field = source, tokenizer, text_field
        self.seq_len, self.packing, self.max_samples = seq_len, packing, max_samples

    def set_epoch(self, epoch: int) -> None:  # lets the Trainer reshuffle HF streaming datasets
        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def __iter__(self):
        examples = iter(self.source)
        if self.max_samples is not None:
            examples = itertools.islice(examples, self.max_samples)
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            examples = itertools.islice(examples, worker.id, None, worker.num_workers)
        eos = self.tokenizer.eos_token_id
        buf: list[int] = []
        for ex in examples:
            ids = self.tokenizer.encode(ex[self.text_field]) + [eos]
            if not self.packing:
                yield {"input_ids": torch.tensor(ids[:self.seq_len])}
                continue
            buf.extend(ids)
            start = 0
            while len(buf) - start >= self.seq_len:
                yield {"input_ids": torch.tensor(buf[start:start + self.seq_len])}
                start += self.seq_len
            buf = buf[start:]


@DATASETS.register_module()
@_objectives("pretrain")
def PretrainDataset(path: str, tokenizer, seq_len: int, name: str | None = None, split: str = "train",
                    text_field: str = "text", packing: bool = True, streaming: bool = False,
                    data_files=None, max_samples: int | None = None, shuffle_buffer: int | None = None,
                    seed: int = 42, num_proc: int | None = None, load_kwargs: dict | None = None):
    """Raw text for next-token prediction. Each document is followed by EOS; with ``packing=True``
    documents are concatenated and cut into blocks of exactly ``seq_len`` tokens (no padding).

    Returns a map-style HF ``Dataset`` (tokenized once, cached by ``datasets``; the Trainer shuffles
    it), or a torch ``IterableDataset`` when ``streaming=True`` (tokenized on the fly, shuffled with a
    ``shuffle_buffer``-sized buffer; the Trainer then needs ``max_steps``).
    """
    if streaming:
        ds = load_stream(path, name, split, data_files, shuffle_buffer, seed, load_kwargs)
        return PackedStream(ds, tokenizer, text_field, seq_len, packing, max_samples)
    ds = load_split(path, name, split, data_files, streaming, max_samples, load_kwargs)

    eos = tokenizer.eos_token_id

    def tokenize(batch):
        return {"input_ids": [ids + [eos] for ids in tokenizer.encode_batch(batch[text_field])]}

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names, num_proc=num_proc, desc="tokenize")
    if packing:
        def pack(batch):  # concatenate a batch of documents and cut it into full blocks
            flat = list(itertools.chain.from_iterable(batch["input_ids"]))
            n = len(flat) // seq_len * seq_len
            return {"input_ids": [flat[i:i + seq_len] for i in range(0, n, seq_len)]}

        ds = ds.map(pack, batched=True, batch_size=1000, num_proc=num_proc, desc=f"pack to {seq_len}")
    else:
        ds = ds.map(lambda b: {"input_ids": [x[:seq_len] for x in b["input_ids"]]}, batched=True,
                    num_proc=num_proc, desc="truncate")
    if len(ds) == 0:
        raise ValueError(f"dataset {path!r} produced no training blocks of seq_len={seq_len}")
    return ds


HFTextDataset = DATASETS.register_module(name="HFTextDataset")(PretrainDataset)  # pre-objectives name


# ------------------------------------------------------------------------------------------------ sft
@DATASETS.register_module()
@_objectives("sft")
def SFTDataset(path: str, tokenizer, max_len: int = 2048, max_prompt_len: int | None = None,
               name: str | None = None, split: str = "train", data_files=None,
               messages_field: str = "messages", prompt_field: str = "prompt", completion_field: str = "completion",
               prompt_template: str | None = None, chat: bool | str = "auto", max_samples: int | None = None,
               num_proc: int | None = None, load_kwargs: dict | None = None):
    """Supervised fine-tuning; the loss covers response tokens only.

    Formats (detected per dataset):
    * conversations: a ``messages_field`` column of ``[{'role', 'content'}, ...]``; every assistant
      turn is trained on (multi-turn supported);
    * prompt / completion: ``prompt_field`` (or ``prompt_template``) and ``completion_field`` columns,
      as plain text or message lists. ``chat=True`` wraps plain-text pairs in the chat template.
    Examples longer than ``max_len`` are truncated; examples left with no response tokens are dropped.
    """
    ds = load_split(path, name, split, data_files, False, max_samples, load_kwargs)
    conversational = messages_field in ds.column_names and prompt_template is None

    def process(row):
        if conversational:
            ids, labels = tokenize_conversation(tokenizer, row[messages_field])
            ids, labels = ids[:max_len], labels[:max_len]
        else:
            p, r = split_prompt_response(tokenizer, _prompt(row, prompt_field, prompt_template),
                                         row[completion_field], chat)
            p, r = _truncate(p, r, max_len, max_prompt_len)
            ids, labels = p + r, [-100] * len(p) + r
        return {"input_ids": ids, "labels": labels}

    ds = ds.map(process, remove_columns=ds.column_names, num_proc=num_proc, desc="tokenize sft")
    ds = ds.filter(lambda row: any(t != -100 for t in row["labels"]), num_proc=num_proc, desc="drop empty")
    if len(ds) == 0:
        raise ValueError(f"SFT dataset {path!r} has no examples with response tokens within max_len={max_len}")
    return ds


# ------------------------------------------------------------------------------------------------ dpo
@DATASETS.register_module()
@_objectives("dpo")
def PreferenceDataset(path: str, tokenizer, max_len: int = 1024, max_prompt_len: int | None = 512,
                      name: str | None = None, split: str = "train", data_files=None,
                      prompt_field: str = "prompt", chosen_field: str = "chosen", rejected_field: str = "rejected",
                      prompt_template: str | None = None, chat: bool | str = "auto",
                      max_samples: int | None = None, num_proc: int | None = None, load_kwargs: dict | None = None):
    """Preference pairs for DPO: a prompt with a chosen and a rejected response.

    Accepts plain text (``prompt``, ``chosen``, ``rejected`` strings) and the chat formats used by
    TRL-style datasets: ``chosen``/``rejected`` as full conversations (``prompt`` optional, e.g.
    ``trl-lib/ultrafeedback_binarized``) or as assistant turns following a ``prompt`` message list.
    """
    ds = load_split(path, name, split, data_files, False, max_samples, load_kwargs)

    def process(row):
        prompt = _prompt(row, prompt_field, prompt_template) if (prompt_template or prompt_field in row) else None
        p, c = split_prompt_response(tokenizer, prompt, row[chosen_field], chat)
        p2, r = split_prompt_response(tokenizer, prompt, row[rejected_field], chat)
        if p2 != p:
            raise ValueError("chosen and rejected responses must share the same prompt")
        p, c = _truncate(p, c, max_len, max_prompt_len)
        _, r = _truncate(p, r, max_len, None)
        return {"prompt_ids": p, "chosen_ids": c, "rejected_ids": r}

    ds = ds.map(process, remove_columns=ds.column_names, num_proc=num_proc, desc="tokenize preferences")
    ds = ds.filter(lambda row: len(row["chosen_ids"]) > 0 and len(row["rejected_ids"]) > 0,
                   num_proc=num_proc, desc="drop empty")
    if len(ds) == 0:
        raise ValueError(f"preference dataset {path!r} has no pairs with response tokens within max_len={max_len}")
    return ds


# ----------------------------------------------------------------------------------------------- grpo
@DATASETS.register_module()
@_objectives("grpo")
def PromptDataset(path: str, tokenizer, max_prompt_len: int = 512, name: str | None = None, split: str = "train",
                  data_files=None, prompt_field: str = "prompt", prompt_template: str | None = None,
                  chat: bool | str = "auto", max_samples: int | None = None, num_proc: int | None = None,
                  load_kwargs: dict | None = None):
    """Prompts for online RL (GRPO). All original columns are kept and passed to the reward
    functions (e.g. a reference ``answer``). Prompts longer than ``max_prompt_len`` are dropped,
    since cutting a question short would make its reward meaningless."""
    ds = load_split(path, name, split, data_files, False, max_samples, load_kwargs)

    def process(row):
        return {"prompt_ids": encode_prompt(tokenizer, _prompt(row, prompt_field, prompt_template), chat)}

    ds = ds.map(process, num_proc=num_proc, desc="tokenize prompts")
    ds = ds.filter(lambda row: 0 < len(row["prompt_ids"]) <= max_prompt_len, num_proc=num_proc, desc="drop long")
    if len(ds) == 0:
        raise ValueError(f"prompt dataset {path!r} has no prompts within max_prompt_len={max_prompt_len}")
    return ds


# ------------------------------------------------------------------------------------------ collators
def _pad(seqs: list[list[int]], labels: list[list[int]], pad_token_id: int) -> dict[str, torch.Tensor]:
    length = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), length), pad_token_id, dtype=torch.long)
    out_labels = torch.full((len(seqs), length), -100, dtype=torch.long)
    for i, (s, lab) in enumerate(zip(seqs, labels)):
        input_ids[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
        out_labels[i, :len(lab)] = torch.as_tensor(lab, dtype=torch.long)
    return {"input_ids": input_ids, "labels": out_labels}


class CausalLMCollator:
    """Pretrain / SFT. Uses each example's ``labels`` if present (SFT), else ``input_ids``.
    Right-pads with ``pad_token_id`` / ``-100``; causal masking makes right padding exact, so no
    attention mask is needed."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [list(map(int, f["input_ids"])) for f in features]
        labels = [list(map(int, f["labels"])) if "labels" in f else s for f, s in zip(features, seqs)]
        batch = _pad(seqs, labels, self.pad_token_id)
        if "source" in features[0]:  # MixturePretrainDataset: which source each row came from (logged)
            batch["source_ids"] = torch.tensor([int(f["source"]) for f in features])
        return batch


class PreferenceCollator:
    """DPO: one ``[2B, L]`` batch, chosen sequences first, then rejected; prompts masked in ``labels``."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs, labels = [], []
        for key in ("chosen_ids", "rejected_ids"):
            for f in features:
                p, r = list(f["prompt_ids"]), list(f[key])
                seqs.append(p + r)
                labels.append([-100] * len(p) + r)
        return _pad(seqs, labels, self.pad_token_id)


class PromptCollator:
    """GRPO: variable-length prompts stay Python lists; every other column is passed through
    (as lists) for the reward functions."""

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        batch = {"prompt_ids": [list(map(int, f["prompt_ids"])) for f in features]}
        batch["columns"] = {k: [f[k] for f in features] for k in features[0] if k != "prompt_ids"}
        return batch


__all__ = ["PretrainDataset", "HFTextDataset", "SFTDataset", "PreferenceDataset", "PromptDataset", "PackedStream",
           "CausalLMCollator", "PreferenceCollator", "PromptCollator", "load_split", "is_messages"]
