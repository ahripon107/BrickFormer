"""Convert Hugging Face Llama checkpoints into an equivalent BrickFormer model (requires ``transformers``)."""

from __future__ import annotations

import re

import torch

from .registry import build_model


def llama_config_from_hf(hf_config) -> dict:
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    rope_params = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
    theta = rope_params.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0))
    rope_kind = rope_params.get("rope_type", rope_params.get("type", "default"))
    scaling = None
    if rope_kind == "llama3":
        scaling = dict(type="llama3", factor=rope_params["factor"], low_freq_factor=rope_params["low_freq_factor"],
                       high_freq_factor=rope_params["high_freq_factor"],
                       original_max_position_embeddings=rope_params["original_max_position_embeddings"])
    elif rope_kind == "linear":
        scaling = dict(type="linear", factor=rope_params["factor"])
    elif rope_kind not in ("default", None):
        raise NotImplementedError(f"RoPE type {rope_kind!r} is not supported by the converter")
    bias = bool(getattr(hf_config, "attention_bias", False))
    return dict(
        type="CausalLM",
        vocab_size=hf_config.vocab_size,
        dim=hf_config.hidden_size,
        pattern=["attn"],
        num_layers=hf_config.num_hidden_layers,
        block_types=dict(attn=dict(
            mixer=dict(type="GQAttention", heads=hf_config.num_attention_heads,
                       kv_heads=hf_config.num_key_value_heads, head_dim=head_dim, bias=bias, out_bias=bias,
                       rope=dict(type="RoPE", theta=theta, scaling=scaling)),
            ffn=dict(type="SwiGLU", hidden=hf_config.intermediate_size,
                     bias=bool(getattr(hf_config, "mlp_bias", False))),
        )),
        norm=dict(type="RMSNorm", eps=hf_config.rms_norm_eps),
        residual="pre_norm",
        tie_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
    )


_KEY_MAP = [
    (r"^model\.embed_tokens\.", "embed."),
    (r"^model\.norm\.", "norm."),
    (r"^lm_head\.", "lm_head."),
    (r"^model\.layers\.(\d+)\.self_attn\.", r"blocks.\1.mixer."),
    (r"^model\.layers\.(\d+)\.mlp\.", r"blocks.\1.ffn."),
    (r"^model\.layers\.(\d+)\.input_layernorm\.", r"blocks.\1.norm1."),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.", r"blocks.\1.norm2."),
]


def convert_llama_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if "rotary_emb" in k:
            continue
        for pat, rep in _KEY_MAP:
            if re.match(pat, k):
                out[re.sub(pat, rep, k)] = v
                break
        else:
            raise KeyError(f"Unmapped HF Llama weight: {k}")
    return out


def from_hf_llama(model_or_path, torch_dtype=None):
    """Build a BrickFormer ``CausalLM`` from an HF ``LlamaForCausalLM`` instance or hub id / path."""
    if isinstance(model_or_path, str):
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(model_or_path, torch_dtype=torch_dtype)
    else:
        hf = model_or_path
    cfg = llama_config_from_hf(hf.config)
    model = build_model(cfg)
    sd = convert_llama_state_dict(hf.state_dict())
    if cfg["tie_embeddings"]:
        sd.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not (cfg["tie_embeddings"] and k == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"HF conversion mismatch. missing={missing} unexpected={unexpected}")
    p = next(hf.parameters())
    return model.to(device=p.device, dtype=p.dtype)
