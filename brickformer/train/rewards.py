"""Reward functions for GRPO, configured by name so experiment configs stay serialisable::

    rewards=[dict(type='NumberMatch', reference_field='answer'),
             dict(type='RegexMatch', pattern=r'####\\s*-?\\d', weight=0.2),
             dict(type='PythonReward', fn='my_pkg.rewards:brevity')]

A reward is called as ``reward(prompts, completions, **columns) -> list[float]`` where ``prompts`` and
``completions`` are decoded strings (one per completion) and ``columns`` holds the dataset's other
columns, repeated to line up with the completions. The total reward is the weighted sum.
"""

from __future__ import annotations

import importlib
import re

from ..registry import REWARDS

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> float | None:
    found = _NUMBER.findall(text or "")
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


class Reward:
    def __init__(self, weight: float = 1.0, name: str | None = None):
        self.weight = weight
        self.name = name or type(self).__name__

    def __call__(self, prompts: list[str], completions: list[str], **columns) -> list[float]:
        raise NotImplementedError


@REWARDS.register_module()
class NumberMatch(Reward):
    """1.0 when the last number in the completion equals the last number in ``reference_field``
    (which handles GSM8K-style ``'... #### 42'`` answers), else 0.0."""

    def __init__(self, reference_field: str = "answer", tolerance: float = 1e-6, **kw):
        super().__init__(**kw)
        self.reference_field, self.tolerance = reference_field, tolerance

    def __call__(self, prompts, completions, **columns):
        refs = columns[self.reference_field]
        out = []
        for completion, ref in zip(completions, refs):
            got, want = last_number(completion), last_number(str(ref))
            out.append(float(got is not None and want is not None and abs(got - want) <= self.tolerance))
        return out


@REWARDS.register_module()
class ExactMatch(Reward):
    """1.0 when the stripped completion (or its ``pattern`` capture group 1) equals ``reference_field``."""

    def __init__(self, reference_field: str = "answer", pattern: str | None = None, ignore_case: bool = False, **kw):
        super().__init__(**kw)
        self.reference_field, self.ignore_case = reference_field, ignore_case
        self.pattern = re.compile(pattern, re.S) if pattern else None

    def _norm(self, text: str) -> str:
        text = text.strip()
        return text.lower() if self.ignore_case else text

    def __call__(self, prompts, completions, **columns):
        out = []
        for completion, ref in zip(completions, columns[self.reference_field]):
            if self.pattern is not None:
                m = self.pattern.search(completion)
                completion = m.group(1) if m else None
            out.append(float(completion is not None and self._norm(completion) == self._norm(str(ref))))
        return out


@REWARDS.register_module()
class RegexMatch(Reward):
    """1.0 when ``pattern`` is found in the completion (useful as a format reward)."""

    def __init__(self, pattern: str, **kw):
        super().__init__(**kw)
        self.pattern = re.compile(pattern, re.S)

    def __call__(self, prompts, completions, **columns):
        return [float(self.pattern.search(c) is not None) for c in completions]


@REWARDS.register_module()
class LengthPenalty(Reward):
    """``-max(0, len(completion) - max_chars) / max_chars``: discourages rambling past ``max_chars``."""

    def __init__(self, max_chars: int = 1000, **kw):
        super().__init__(**kw)
        self.max_chars = max_chars

    def __call__(self, prompts, completions, **columns):
        return [-max(0, len(c) - self.max_chars) / self.max_chars for c in completions]


@REWARDS.register_module()
class PythonReward(Reward):
    """Any function given as ``'package.module:function'`` with the reward call signature; extra
    keyword arguments in the config are forwarded to it."""

    def __init__(self, fn: str, weight: float = 1.0, name: str | None = None, **kwargs):
        module, _, attr = fn.partition(":")
        if not attr:
            raise ValueError(f"PythonReward fn must look like 'package.module:function', got {fn!r}")
        self.fn = getattr(importlib.import_module(module), attr)
        self.kwargs = kwargs
        super().__init__(weight=weight, name=name or attr)

    def __call__(self, prompts, completions, **columns):
        return [float(x) for x in self.fn(prompts, completions, **columns, **self.kwargs)]
