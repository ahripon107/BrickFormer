"""Import Hugging Face checkpoints (Llama, Mistral, Qwen2, Qwen3) into BrickFormer (requires ``transformers``).

* ``from_hf(model_or_path)`` builds the equivalent BrickFormer model and loads its weights.
* ``load_hf_weights(model, path)`` loads a checkpoint into a model built from *your* config, after
  checking that the config describes the same architecture (``architecture_diff``).
* ``config_from_hf(hf_config)`` gives the BrickFormer model config for an HF config.

Weights are read straight from the ``*.safetensors`` (or ``*.bin``) files, so the HF model class is
never instantiated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from .registry import build_model
from .utils.arch import ArchitectureMismatch, architecture_diff

SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3")


def _rope_from_hf(hf_config) -> dict:
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
    partial = rope_params.get("partial_rotary_factor", getattr(hf_config, "partial_rotary_factor", 1.0))
    if partial not in (None, 1, 1.0):
        raise NotImplementedError(f"partial_rotary_factor={partial} is not supported by the converter")
    return dict(type="RoPE", theta=theta, scaling=scaling)


def _layer_windows(hf_config) -> list[int | None]:
    """Sliding-window size per layer (``None`` = full attention)."""
    n, window = hf_config.num_hidden_layers, getattr(hf_config, "sliding_window", None)
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:  # Mistral: one window for every layer
        return [window] * n
    unknown = set(layer_types) - {"full_attention", "sliding_attention"}
    if unknown:
        raise NotImplementedError(f"layer types {sorted(unknown)} are not supported by the converter")
    return [window if t == "sliding_attention" else None for t in layer_types]


def config_from_hf(hf_config) -> dict:
    """BrickFormer ``CausalLM`` config equivalent to an HF Llama / Mistral / Qwen2 / Qwen3 config."""
    model_type = getattr(hf_config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(f"HF model_type {model_type!r} is not supported; "
                                  f"supported: {', '.join(SUPPORTED_MODEL_TYPES)}")
    heads = hf_config.num_attention_heads
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // heads
    if model_type == "qwen2":  # bias on q/k/v only
        bias, out_bias = True, False
    else:
        bias = out_bias = bool(getattr(hf_config, "attention_bias", False))
    mixer = dict(type="GQAttention", heads=heads, kv_heads=getattr(hf_config, "num_key_value_heads", None) or heads,
                 head_dim=head_dim, bias=bias, out_bias=out_bias, rope=_rope_from_hf(hf_config))
    if model_type == "qwen3":
        mixer.update(qk_norm=True, norm_eps=hf_config.rms_norm_eps)
    ffn = dict(type="SwiGLU", hidden=hf_config.intermediate_size, bias=bool(getattr(hf_config, "mlp_bias", False)))

    windows = _layer_windows(hf_config)
    block_types, pattern = {}, []
    for w in windows:
        name = "attn" if w is None else "swa"
        if name not in block_types:
            m = dict(mixer) if w is None else dict(mixer, type="SlidingWindowAttention", window=w)
            block_types[name] = dict(mixer=m, ffn=dict(ffn))
        pattern.append(name)
    if len(set(pattern)) == 1:
        pattern, num_layers = [pattern[0]], len(pattern)
    else:
        num_layers = None
    return dict(
        type="CausalLM",
        vocab_size=hf_config.vocab_size,
        dim=hf_config.hidden_size,
        pattern=pattern,
        num_layers=num_layers,
        block_types=block_types,
        norm=dict(type="RMSNorm", eps=hf_config.rms_norm_eps),
        residual="pre_norm",
        tie_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
    )


llama_config_from_hf = config_from_hf  # backward compatible name

_KEY_MAP = [
    (r"^model\.embed_tokens\.", "embed."),
    (r"^model\.norm\.", "norm."),
    (r"^lm_head\.", "lm_head."),
    (r"^model\.layers\.(\d+)\.self_attn\.", r"blocks.\1.mixer."),  # q/k/v/o_proj, q_norm, k_norm
    (r"^model\.layers\.(\d+)\.mlp\.", r"blocks.\1.ffn."),
    (r"^model\.layers\.(\d+)\.input_layernorm\.", r"blocks.\1.norm1."),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.", r"blocks.\1.norm2."),
]


def convert_hf_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if "rotary_emb" in k:
            continue
        for pat, rep in _KEY_MAP:
            if re.match(pat, k):
                out[re.sub(pat, rep, k)] = v
                break
        else:
            raise KeyError(f"Unmapped HF weight: {k}")
    return out


convert_llama_state_dict = convert_hf_state_dict  # backward compatible name


def load_hf_config(path: str, revision: str | None = None):
    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise ImportError("importing HF checkpoints requires `pip install transformers`") from e
    return AutoConfig.from_pretrained(path, revision=revision)


def _weight_files(path: str, revision: str | None = None) -> list[Path]:
    local = Path(path)
    if not local.is_dir():
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(path, revision=revision,
                                       allow_patterns=["*.safetensors", "*.safetensors.index.json"]))
        if not list(local.glob("*.safetensors")):
            local = Path(snapshot_download(path, revision=revision, allow_patterns=["*.bin", "*.bin.index.json"]))
    files = sorted(local.glob("*.safetensors")) or sorted(local.glob("*.bin"))
    index = next(iter(sorted(local.glob("*.index.json"))), None)
    if index is not None:  # sharded: only the files the index names (skips stray consolidated copies)
        names = set(json.loads(index.read_text())["weight_map"].values())
        files = [f for f in files if f.name in names]
    if not files:
        raise FileNotFoundError(f"no *.safetensors or *.bin weights found for {path!r}")
    return files


def load_hf_state_dict(path: str, revision: str | None = None) -> dict[str, torch.Tensor]:
    """The raw HF state dict of a Hub repo id or local directory (CPU tensors, checkpoint dtype)."""
    sd = {}
    for f in _weight_files(path, revision):
        if f.suffix == ".safetensors":
            from safetensors.torch import load_file

            sd.update(load_file(str(f)))
        else:
            sd.update(torch.load(f, map_location="cpu", weights_only=True))
    return sd


def _load_converted(model, sd: dict[str, torch.Tensor], tied: bool) -> None:
    sd = convert_hf_state_dict(sd)
    if tied:
        sd.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not (tied and k == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"HF conversion mismatch. missing={missing} unexpected={unexpected}")


def load_hf_weights(model, path: str, revision: str | None = None, strict: bool = True, hf_config=None):
    """Load an HF checkpoint (Hub id or local directory) into a BrickFormer ``model`` built from your config.

    ``strict=True`` first checks that ``model`` has exactly the checkpoint's architecture and raises
    ``ArchitectureMismatch`` listing every difference. ``strict=False`` only requires the parameter
    names and shapes to line up, so behavioural settings (RoPE theta / scaling, norm eps, sliding
    windows) may deliberately differ, e.g. to extend the context. Returns ``model``.
    """
    hf_config = hf_config if hf_config is not None else load_hf_config(path, revision)
    reference = config_from_hf(hf_config)
    if strict:
        diffs = architecture_diff(model, reference, "config", "pretrained")
        if diffs:
            raise ArchitectureMismatch(
                f"model config does not match the architecture of {path!r} "
                f"({len(diffs)} difference{'s' * (len(diffs) > 1)}):\n  " + "\n  ".join(diffs)
                + "\nFix the model config, omit it to use the checkpoint's architecture, or pass strict=False "
                  "to allow non-weight differences.")
    tied = model.lm_head.weight is model.embed.weight
    if tied != reference["tie_embeddings"]:
        raise ArchitectureMismatch(f"tie_embeddings: config={tied}, pretrained={reference['tie_embeddings']}")
    _load_converted(model, load_hf_state_dict(path, revision), tied)
    return model


def from_hf(model_or_path, torch_dtype=None, revision: str | None = None):
    """Build a BrickFormer ``CausalLM`` from an HF model instance, Hub id or local directory.

    For a path the model is built from the checkpoint's ``config.json`` and the weights are read from
    its files; it is returned in float32 unless ``torch_dtype`` is given. An HF model instance keeps
    its device and dtype.
    """
    if isinstance(model_or_path, (str, Path)):
        path = str(model_or_path)
        hf_config = load_hf_config(path, revision)
        model = build_model(config_from_hf(hf_config))
        load_hf_weights(model, path, revision, strict=False, hf_config=hf_config)
        return model.to(torch_dtype) if torch_dtype is not None else model
    hf = model_or_path
    cfg = config_from_hf(hf.config)
    p = next(hf.parameters())
    model = build_model(cfg).to(dtype=p.dtype)  # cast before loading so no precision is lost on the way
    _load_converted(model, hf.state_dict(), cfg["tie_embeddings"])
    return model.to(device=p.device)


from_hf_llama = from_hf  # backward compatible name
