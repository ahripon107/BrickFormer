"""Importing HF Qwen2 / Qwen3 / Mistral checkpoints, architecture checks, and ``pretrained=`` in experiments."""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

import brickformer as bf  # noqa: E402
from brickformer.hf import config_from_hf, from_hf, load_hf_weights  # noqa: E402

BASE = dict(vocab_size=320, hidden_size=64, intermediate_size=160, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128, rms_norm_eps=1e-6, attn_implementation="sdpa")


def _hf(kind, **kw):
    cfg = dict(BASE, **kw)
    conf = getattr(transformers, f"{kind}Config")(**cfg)
    return getattr(transformers, f"{kind}ForCausalLM")(conf).double().eval()


CASES = {
    "qwen3_tied": ("Qwen3", dict(head_dim=32, tie_word_embeddings=True, rope_theta=1e6)),  # head_dim != dim/heads
    "qwen2_bias": ("Qwen2", dict(tie_word_embeddings=False)),  # q/k/v bias, no o_proj bias
    "qwen2_mixed_swa": ("Qwen2", dict(use_sliding_window=True, sliding_window=8, max_window_layers=1)),
    "mistral_swa": ("Mistral", dict(sliding_window=8)),
}


@pytest.mark.parametrize("case", CASES)
def test_logits_match(case):
    torch.manual_seed(0)
    kind, kw = CASES[case]
    hf = _hf(kind, **kw)
    with torch.no_grad():  # random q/k norm weights so QK-norm is actually exercised
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    ours = from_hf(hf).eval()
    ids = torch.randint(0, 320, (2, 40))  # longer than the sliding windows
    with torch.no_grad():
        torch.testing.assert_close(ours(ids).logits, hf(ids).logits, atol=1e-10, rtol=1e-8)


def test_mixed_sliding_window_config():
    kind, kw = CASES["qwen2_mixed_swa"]
    hf = _hf(kind, **kw)
    cfg = config_from_hf(hf.config)
    windows = [hf.config.sliding_window if t == "sliding_attention" else None for t in hf.config.layer_types]
    assert len(set(windows)) == 2, "test needs both layer kinds"
    assert cfg["pattern"] == ["attn" if w is None else "swa" for w in windows]
    assert cfg["block_types"]["swa"]["mixer"]["window"] == 8


def test_from_path_reads_safetensors(tmp_path):
    torch.manual_seed(0)
    hf = _hf("Qwen3", head_dim=32, tie_word_embeddings=True)
    hf.save_pretrained(tmp_path)
    ours = from_hf(str(tmp_path), torch_dtype=torch.float64).eval()
    ids = torch.randint(0, 320, (1, 20))
    with torch.no_grad():
        torch.testing.assert_close(ours(ids).logits, hf(ids).logits, atol=1e-10, rtol=1e-8)


def test_unsupported_model_type():
    conf = transformers.GPT2Config(n_layer=1, n_embd=32, n_head=2)
    with pytest.raises(NotImplementedError, match="qwen3"):
        config_from_hf(conf)


# ---------------------------------------------------------------- architecture checks
def _qwen3_cfg():
    return dict(type="CausalLM", vocab_size=320, dim=64, pattern="A", num_layers=3, tie_embeddings=True,
                block_types=dict(attn=dict(
                    mixer=dict(type="GQAttention", heads=4, kv_heads=2, head_dim=32, qk_norm=True,
                               rope=dict(type="RoPE", theta=1e6)),
                    ffn=dict(type="SwiGLU", hidden=160))),
                norm=dict(type="RMSNorm", eps=1e-6), grad_checkpointing=True)


def test_architecture_diff_equivalent_configs():
    hf = _hf("Qwen3", head_dim=32, tie_word_embeddings=True, rope_theta=1e6)
    # different pattern spelling, block name and training-only settings: still the same architecture
    assert bf.architecture_diff(_qwen3_cfg(), config_from_hf(hf.config)) == []
    mha = dict(type="CausalLM", vocab_size=50, dim=32, pattern="A", num_layers=2,
               block_types=dict(attn=dict(mixer=dict(type="MHAttention", heads=4))))
    gqa = copy.deepcopy(mha)
    gqa["block_types"]["attn"]["mixer"] = dict(type="GQAttention", heads=4, kv_heads=4)
    assert bf.architecture_diff(mha, gqa) == []


def test_architecture_diff_reports_each_difference():
    ref = _qwen3_cfg()
    bad = copy.deepcopy(ref)
    bad["block_types"]["attn"]["mixer"].update(qk_norm=False, rope=dict(type="RoPE", theta=1e4))
    bad["norm"]["eps"] = 1e-5
    bad["num_layers"] = 4
    bad["tie_embeddings"] = False
    diffs = bf.architecture_diff(bad, ref)
    text = "\n".join(diffs)
    assert "num_layers: config=4, pretrained=3" in text
    assert "tie_embeddings: config=False, pretrained=True" in text
    assert "blocks.[0-2].mixer.q_norm: only in pretrained" in text
    assert "blocks.[3]: only in config" in text
    assert "blocks.[0-2].mixer.rope.theta: config=10000.0, pretrained=1000000.0" in text
    assert "blocks.[0-2].norm1.eps: config=1e-05, pretrained=1e-06" in text
    assert "norm.eps" in text  # final norm


def test_load_hf_weights_strict_and_relaxed(tmp_path):
    torch.manual_seed(0)
    hf = _hf("Qwen3", head_dim=32, tie_word_embeddings=True, rope_theta=1e6)
    hf.save_pretrained(tmp_path)
    model = load_hf_weights(bf.build_model(_qwen3_cfg()), str(tmp_path))
    torch.testing.assert_close(model.blocks[1].mixer.q_proj.weight.double(),
                               hf.model.layers[1].self_attn.q_proj.weight)

    longer_ctx = _qwen3_cfg()
    longer_ctx["block_types"]["attn"]["mixer"]["rope"]["scaling"] = dict(type="linear", factor=4.0)
    with pytest.raises(bf.ArchitectureMismatch, match="rope.scaling"):
        load_hf_weights(bf.build_model(longer_ctx), str(tmp_path))
    load_hf_weights(bf.build_model(longer_ctx), str(tmp_path), strict=False)  # same weights, allowed

    wrong_shape = _qwen3_cfg()
    wrong_shape["block_types"]["attn"]["mixer"]["head_dim"] = 16
    with pytest.raises(RuntimeError, match="size mismatch"):  # strict=False still needs matching shapes
        load_hf_weights(bf.build_model(wrong_shape), str(tmp_path), strict=False)


# ---------------------------------------------------------------- experiments
@pytest.fixture
def hf_dir(tmp_path):
    torch.manual_seed(0)
    hf = _hf("Qwen3", head_dim=32, tie_word_embeddings=True, rope_theta=1e6)
    hf.save_pretrained(tmp_path / "hf")
    return str(tmp_path / "hf"), hf


def _exp_cfg(tmp_path, **kw):
    cfg = dict(tokenizer=dict(type="ByteTokenizer"),
               data=dict(train=dict(path="text", data_files="unused.txt", seq_len=32)),
               train=dict(type="HFTrainer", output_dir=str(tmp_path / "run"), max_steps=1, use_cpu=True))
    cfg.update(kw)
    return cfg


def test_experiment_pretrained_with_matching_config(tmp_path, hf_dir):
    pytest.importorskip("datasets")
    from brickformer.train import build_experiment

    path, hf = hf_dir
    model_cfg = _qwen3_cfg()
    model_cfg["vocab_size"] = "auto"  # resolved to the checkpoint's vocabulary, not the tokenizer's
    exp = build_experiment(_exp_cfg(tmp_path, pretrained=path, model=model_cfg), build_data=False)
    assert exp.config["model"]["vocab_size"] == 320
    assert exp.model.grad_checkpointing
    ids = torch.randint(0, 257, (1, 16))
    with torch.no_grad():
        torch.testing.assert_close(exp.model.double().eval()(ids).logits, hf(ids).logits, atol=1e-10, rtol=1e-8)


def test_experiment_pretrained_mismatch_raises(tmp_path, hf_dir):
    pytest.importorskip("datasets")
    from brickformer.train import build_experiment

    bad = _qwen3_cfg()
    bad["block_types"]["attn"]["mixer"]["qk_norm"] = False
    with pytest.raises(bf.ArchitectureMismatch, match=r"q_norm: only in pretrained"):
        build_experiment(_exp_cfg(tmp_path, pretrained=hf_dir[0], model=bad), build_data=False)
    with pytest.raises(ValueError, match="not both"):
        build_experiment(_exp_cfg(tmp_path, pretrained=hf_dir[0], init_from=str(tmp_path)), build_data=False)


def test_experiment_pretrained_without_model_section(tmp_path, hf_dir):
    pytest.importorskip("datasets")
    from brickformer.train import build_experiment

    path, hf = hf_dir
    # no model section: the checkpoint's architecture; a partial section only adds settings
    exp = build_experiment(_exp_cfg(tmp_path, pretrained=dict(path=path), model=dict(grad_checkpointing=True)),
                           build_data=False)
    assert exp.model.grad_checkpointing
    assert bf.architecture_diff(exp.model, config_from_hf(hf.config)) == []
    exp = build_experiment(_exp_cfg(tmp_path, pretrained=path), build_data=False)
    assert exp.model.num_params() == sum(p.numel() for p in hf.parameters())
