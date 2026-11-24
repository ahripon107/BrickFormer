import pytest
import torch

# Tiny versions of every mixer, all with dim=64, used across tests.
DIM = 64
MIXER_CFGS = {
    "mha": dict(type="MHAttention", heads=4),
    "gqa": dict(type="GQAttention", heads=4, kv_heads=2),
    "gqa_qknorm": dict(type="GQAttention", heads=4, kv_heads=1, qk_norm=True,
                       rope=dict(type="RoPE", theta=1e4, scaling=dict(type="llama3", factor=8.0,
                                 low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=16))),
    "swa": dict(type="SlidingWindowAttention", heads=4, kv_heads=2, window=5),
    "mla": dict(type="MLA", heads=4, kv_lora_rank=32, q_lora_rank=24, nope_head_dim=16, rope_head_dim=8,
                v_head_dim=12),
    "mamba2": dict(type="Mamba2", d_state=16, headdim=16, expand=2, ngroups=2, chunk_size=8, backend="torch"),
    "gdn": dict(type="GatedDeltaNet", heads=4, head_dim=16, expand_v=1.5, chunk_size=8, backend="torch"),
}


def tiny_lm(pattern, block_types=None, **kw):
    import brickformer as bf

    block_types = block_types or {k: dict(mixer=v, ffn=dict(type="SwiGLU", hidden=128)) for k, v in MIXER_CFGS.items()}
    cfg = dict(type="CausalLM", vocab_size=97, dim=DIM, pattern=pattern, block_types=block_types, **kw)
    return bf.build_model(cfg)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
