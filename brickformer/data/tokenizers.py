"""Tokenizer adapters. Every tokenizer exposes ``vocab_size``, ``eos_token_id``, ``pad_token_id``,
``encode``, ``encode_batch``, ``decode``, ``render_chat`` and ``save_pretrained``."""

from __future__ import annotations

import json
from pathlib import Path

from ..registry import TOKENIZERS
from .chat import default_chat_template


@TOKENIZERS.register_module()
class ByteTokenizer:
    """UTF-8 bytes (0-255) plus an EOS token (256). Needs no downloads; good for tests and tiny runs."""

    def __init__(self):
        self.vocab_size = 257
        self.eos_token_id = 256
        self.pad_token_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def render_chat(self, messages, add_generation_prompt: bool = False) -> str:
        return default_chat_template(messages, add_generation_prompt)

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory, "tokenizer_config.json").write_text(json.dumps(dict(type="ByteTokenizer")))


@TOKENIZERS.register_module()
class HFTokenizer:
    """Wraps ``transformers.AutoTokenizer.from_pretrained(name, **kwargs)``.

    Text is encoded without special tokens; documents are separated by EOS when packing.
    ``chat_template``: a Jinja template overriding the tokenizer's own. Tokenizers without one
    (e.g. GPT-2) fall back to a ChatML-style template.
    """

    def __init__(self, name: str, chat_template: str | None = None, **kwargs):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("HFTokenizer requires `pip install transformers` "
                              "(or `pip install brickformer[train]`)") from e
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, **kwargs)
        if chat_template is not None:
            self.tok.chat_template = chat_template
        self.tok.model_max_length = int(1e30)  # we chunk ourselves; silence length warnings
        if self.tok.eos_token_id is None:
            raise ValueError(f"tokenizer {name!r} has no EOS token, which is needed to separate documents")
        self.vocab_size = len(self.tok)
        self.eos_token_id = self.tok.eos_token_id
        self.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.tok(texts, add_special_tokens=False)["input_ids"]

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def render_chat(self, messages, add_generation_prompt: bool = False) -> str:
        if not self.tok.chat_template:
            return default_chat_template(messages, add_generation_prompt)
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def save_pretrained(self, directory: str | Path) -> None:
        self.tok.save_pretrained(directory)
