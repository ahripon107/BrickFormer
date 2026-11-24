"""Parallel forward over a full sequence must equal (a) token-by-token decoding with state and
(b) chunked prefill followed by decoding, for every mixer and for a hybrid model."""

import pytest
import torch
from conftest import DIM, MIXER_CFGS, tiny_lm

import brickformer as bf

T = 21  # deliberately not a multiple of the chunk sizes


@pytest.mark.parametrize("name", list(MIXER_CFGS))
def test_mixer_parallel_vs_stepwise(name):
    mixer = bf.MIXERS.build(MIXER_CFGS[name], dim=DIM).double()
    x = torch.randn(2, T, DIM, dtype=torch.float64)
    y_par, _ = mixer(x)

    state = mixer.init_state(2, T, dtype=torch.float64)
    steps = []
    for t in range(T):
        y, state = mixer(x[:, t:t + 1], state)
        steps.append(y)
    torch.testing.assert_close(torch.cat(steps, 1), y_par, atol=1e-9, rtol=1e-7)

    # chunked prefill (13 tokens) then 2 tokens at once, then single steps
    state = mixer.init_state(2, T, dtype=torch.float64)
    y1, state = mixer(x[:, :13], state)
    y2, state = mixer(x[:, 13:15], state)
    rest = []
    for t in range(15, T):
        y, state = mixer(x[:, t:t + 1], state)
        rest.append(y)
    torch.testing.assert_close(torch.cat([y1, y2, *rest], 1), y_par, atol=1e-9, rtol=1e-7)


def test_hybrid_model_cache():
    model = tiny_lm(list(MIXER_CFGS)).double().eval()
    ids = torch.randint(0, 97, (2, T))
    full = model(ids).logits
    cache = model.init_cache(2, T)
    pre = model(ids[:, :9], cache=cache).logits
    steps = [model(ids[:, t:t + 1], cache=cache).logits for t in range(9, T)]
    torch.testing.assert_close(torch.cat([pre, *steps], 1), full, atol=1e-8, rtol=1e-6)
    assert cache.seq_len == T


def test_kv_cache_overflow():
    mixer = bf.MIXERS.build(MIXER_CFGS["gqa"], dim=DIM)
    state = mixer.init_state(1, 4)
    with pytest.raises(RuntimeError, match="overflow"):
        mixer(torch.randn(1, 5, DIM), state)
