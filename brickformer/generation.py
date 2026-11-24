"""Autoregressive generation over any mix of mixer caches (KV, latent KV, SSM / recurrent states)."""

from __future__ import annotations

import torch


def sample_next(logits: torch.Tensor, temperature: float = 0.0, top_k: int | None = None,
                top_p: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """``temperature == 0`` means greedy. ``logits``: [B, V] -> [B] token ids."""
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, -torch.inf)
    if top_p < 1.0:
        sorted_logits, idx = logits.sort(dim=-1, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        drop = cum - sorted_logits.softmax(-1) > top_p  # keep the smallest prefix with mass >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, idx, sorted_logits)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
             top_k: int | None = None, top_p: float = 1.0, eos_token_id: int | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Prefill the prompt in one parallel pass, then decode token by token with per-layer state.

    Prompts in a batch must have equal length (no padding). Returns ``[B, T + new]``; rows that
    hit ``eos_token_id`` are padded with it until all rows finish.
    """
    b, t = input_ids.shape
    cache = model.init_cache(b, t + max_new_tokens)
    logits = model(input_ids, cache=cache).logits[:, -1]
    finished = torch.zeros(b, dtype=torch.bool, device=input_ids.device)
    out = [input_ids]
    for step in range(max_new_tokens):
        nxt = sample_next(logits, temperature, top_k, top_p, generator)
        if eos_token_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
            finished |= nxt == eos_token_id
        out.append(nxt[:, None])
        if (eos_token_id is not None and bool(finished.all())) or step == max_new_tokens - 1:
            break
        logits = model(nxt[:, None], cache=cache).logits[:, -1]
    return torch.cat(out, dim=1)
