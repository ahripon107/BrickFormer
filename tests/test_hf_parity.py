"""Llama-style BrickFormer model must reproduce HF LlamaForCausalLM logits exactly (fp32/fp64)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from brickformer.hf import from_hf_llama  # noqa: E402


def _hf_llama(**kw):
    cfg = dict(vocab_size=101, hidden_size=64, intermediate_size=160, num_hidden_layers=3,
               num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
               rms_norm_eps=1e-5, tie_word_embeddings=False, attn_implementation="sdpa")
    cfg.update(kw)
    return transformers.LlamaForCausalLM(transformers.LlamaConfig(**cfg)).double().eval()


@pytest.mark.parametrize("kw", [
    {},
    dict(num_key_value_heads=4, tie_word_embeddings=True),
    dict(rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                           original_max_position_embeddings=32), rope_theta=500000.0),
])
def test_llama_logits_match(kw):
    torch.manual_seed(0)
    hf = _hf_llama(**kw)
    ours = from_hf_llama(hf).eval()
    ids = torch.randint(0, 101, (2, 40))
    with torch.no_grad():
        ref = hf(ids).logits
        out = ours(ids).logits
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-8)


def test_llama_generation_matches():
    torch.manual_seed(0)
    hf = _hf_llama()
    ours = from_hf_llama(hf).eval()
    ids = torch.randint(0, 101, (1, 7))
    ref = hf.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=0)
    out = ours.generate(ids, max_new_tokens=12)
    assert torch.equal(out, ref)
