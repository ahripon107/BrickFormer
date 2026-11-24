from pathlib import Path

import pytest
import torch
from conftest import DIM, MIXER_CFGS, tiny_lm

import brickformer as bf
from brickformer.block import RESIDUAL_STYLES
from brickformer.models import expand_pattern

CONFIGS = Path(__file__).parent.parent / "configs"


# ---------------------------------------------------------------- registry / patterns
def test_registry_build_and_errors():
    reg = bf.Registry("t")

    @reg.register_module()
    class Foo:
        def __init__(self, a, b=1):
            self.a, self.b = a, b

    foo = reg.build(dict(type="Foo", a=3), b=5)
    assert (foo.a, foo.b) == (3, 5)
    assert reg.build(dict(type="Foo", a=3, b=2), b=5).b == 2  # explicit beats default
    with pytest.raises(KeyError, match="already registered"):
        reg.register_module()(Foo)
    with pytest.raises(KeyError, match="not registered"):
        reg.build(dict(type="Bar"))
    with pytest.raises(TypeError):
        reg.build(dict(a=1))


def test_pattern_expansion():
    bt = dict(attn={}, gdn={}, x={})
    assert expand_pattern("GGG*", bt, num_layers=8) == ["gdn", "gdn", "gdn", "attn"] * 2
    assert expand_pattern("xG-x", bt) == ["x", "gdn", "x"]
    assert expand_pattern("QA", bt, pattern_chars=dict(Q="x")) == ["x", "attn"]
    with pytest.raises(KeyError):
        expand_pattern("M", bt)  # 'mamba' not defined
    with pytest.raises(ValueError):
        expand_pattern("GG*", bt, num_layers=4)


# ---------------------------------------------------------------- hybrid smoke test
def test_hybrid_forward_backward_generate():
    dense = {k: dict(mixer=v, ffn=dict(type="SwiGLU", hidden=96)) for k, v in MIXER_CFGS.items()}
    model = tiny_lm("mha gqa swa mla mamba2 gdn".split(), pattern_chars=None,
                    block_types={**dense,
                                 "mamba2": dict(mixer=MIXER_CFGS["mamba2"], ffn=None),
                                 "gdn": dict(mixer=MIXER_CFGS["gdn"],
                                             ffn=dict(type="MoE", num_experts=4, top_k=2, hidden=32))},
                    tie_embeddings=True)
    ids = torch.randint(0, 97, (3, 33))
    out = model(ids, labels=ids)
    assert out.logits.shape == (3, 33, 97)
    assert out.aux_loss is not None and out.aux_loss > 0
    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    model.eval()
    gen = model.generate(ids[:, :5], max_new_tokens=10, temperature=0.8, top_p=0.9, top_k=20)
    assert gen.shape == (3, 15) and torch.equal(gen[:, :5], ids[:, :5])
    greedy = bf.generate(model, ids[:, :5], 6)
    assert torch.equal(greedy, bf.generate(model, ids[:, :5], 6))


def test_generate_eos_stops():
    model = tiny_lm(["gqa"]).eval()
    ids = torch.randint(0, 97, (2, 4))
    first = model.generate(ids, 1)[:, -1]
    out = model.generate(ids, 10, eos_token_id=int(first[0]))
    assert out.shape[1] <= 14 and (out[0, 4:] == first[0]).all()


@pytest.mark.parametrize("residual", RESIDUAL_STYLES)
@pytest.mark.parametrize("ffn", [True, False])
def test_residual_styles(residual, ffn):
    bt = dict(a=dict(mixer=MIXER_CFGS["gqa"], ffn=dict(type="GeLUMLP") if ffn else None))
    model = tiny_lm("aa", block_types=bt, residual=residual, norm=dict(type="LayerNorm"),
                    final_norm=residual != "post_norm")
    ids = torch.randint(0, 97, (2, 9))
    model(ids, labels=ids).loss.backward()
    model.eval()
    cache = model.init_cache(2, 9)
    step = torch.cat([model(ids[:, i:i + 1], cache=cache).logits for i in range(9)], 1)
    torch.testing.assert_close(step, model(ids).logits, atol=1e-5, rtol=1e-4)


def test_key_padding_mask_left_padding():
    """Left-padded sequences must give the same logits on real tokens as unpadded ones."""
    model = tiny_lm(["gqa", "mamba2", "gdn", "mla"]).double().eval()
    ids = torch.randint(1, 97, (1, 10))
    ref = model(ids).logits
    padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), ids], 1)
    mask = torch.cat([torch.zeros(1, 3, dtype=torch.bool), torch.ones(1, 10, dtype=torch.bool)], 1)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(padded, attention_mask=mask, position_ids=pos).logits[:, 3:]
    torch.testing.assert_close(out, ref, atol=1e-8, rtol=1e-6)


def test_grad_checkpointing_same_grads():
    m1 = tiny_lm(["gqa", "gdn"])
    m2 = tiny_lm(["gqa", "gdn"], grad_checkpointing=True)
    m2.load_state_dict(m1.state_dict())
    ids = torch.randint(0, 97, (2, 12))
    m1(ids, labels=ids).loss.backward()
    m2(ids, labels=ids).loss.backward()
    for (n, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, msg=n)


# ---------------------------------------------------------------- MoE
def test_moe_routing_and_aux_loss():
    moe = bf.FFNS.build(dict(type="MoE", num_experts=4, top_k=2, hidden=16, shared_experts=1), dim=DIM)
    x = torch.randn(2, 7, DIM, requires_grad=True)
    y = moe(x)
    assert y.shape == x.shape
    # balance loss is >= coef (equality iff perfectly balanced)
    assert moe.aux_loss >= moe.aux_loss_coef * 0.999
    (y.sum() + moe.aux_loss).backward()
    assert moe.router.weight.grad is not None and moe.w_gate.grad.abs().sum() > 0

    # top_k == num_experts with normalised probs == dense softmax mixture of all experts
    moe = bf.FFNS.build(dict(type="MoE", num_experts=3, top_k=3, hidden=8), dim=DIM)
    x = torch.randn(5, DIM)
    p = moe.router(x).softmax(-1)
    import torch.nn.functional as F
    dense = sum(p[:, e, None] * ((F.silu(x @ moe.w_gate[e]) * (x @ moe.w_up[e])) @ moe.w_down[e]) for e in range(3))
    torch.testing.assert_close(moe(x), dense, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- configs
@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.*")), ids=lambda p: p.name)
def test_example_configs_build_and_roundtrip(path, tmp_path):
    model = bf.build_from_file(path)
    assert len(model.blocks) > 0
    text = bf.summary(model, seq_len=1024)
    assert "total" in text
    for suffix in (".yaml", ".json", ".py"):
        f = tmp_path / f"cfg{suffix}"
        bf.dump_config(model.config, f)
        rebuilt = bf.build_from_file(f)
        assert rebuilt.layer_types == model.layer_types
        assert str(rebuilt) == str(model)
        shapes = {k: v.shape for k, v in model.state_dict().items()}
        assert {k: v.shape for k, v in rebuilt.state_dict().items()} == shapes


def test_save_load_pretrained(tmp_path):
    model = tiny_lm(["gqa", "mamba2", "mla"], tie_embeddings=True).eval()
    bf.save_pretrained(model, tmp_path)
    loaded = bf.load_pretrained(tmp_path).eval()
    ids = torch.randint(0, 97, (1, 8))
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits)
    assert loaded.lm_head.weight is loaded.embed.weight


def test_param_and_flop_counts():
    model = tiny_lm(["gqa", "gqa", "gdn"], tie_embeddings=True)
    p = bf.count_params(model)
    assert p["gqa"]["layers"] == 2 and p["gdn"]["layers"] == 1
    assert p["total"] == sum(x.numel() for x in model.parameters())
    assert p["gqa"]["total"] + p["gdn"]["total"] + p["embed"]["total"] + DIM == p["total"]  # + final norm
    f1, f2 = bf.estimate_flops(model, 128), bf.estimate_flops(model, 1024)
    assert f2["per_type"]["gqa"] > f1["per_type"]["gqa"]  # attention grows with context
    assert f2["per_type"]["gdn"] == f1["per_type"]["gdn"]  # recurrent cost is constant
