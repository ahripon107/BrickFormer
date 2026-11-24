# BrickFormer

Build LLMs from mixed block types, defined entirely in config, the way mmdetection works for
detectors. Every component is a registered module picked by name (`dict(type='GQAttention', ...)`),
so you can compare attention, sliding-window attention, MLA, Mamba2 and Gated DeltaNet with
dense or MoE feed-forwards without touching model code.

```python
import brickformer as bf

model = bf.build_model(dict(
    type='CausalLM', vocab_size=32000, dim=1024,
    pattern='GGG*', num_layers=24,                 # = ['gdn', 'gdn', 'gdn', 'attn'] * 6
    block_types=dict(
        attn=dict(mixer=dict(type='GQAttention', heads=16, kv_heads=4, rope=True),
                  ffn=dict(type='SwiGLU', hidden=2816)),
        gdn=dict(mixer=dict(type='GatedDeltaNet', heads=8),
                 ffn=dict(type='MoE', num_experts=8, top_k=2, hidden=1408)),
    ),
    norm=dict(type='RMSNorm'), residual='pre_norm', tie_embeddings=True,
))

out = model(input_ids, labels=input_ids)          # out.loss includes the MoE aux loss in training
tokens = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_p=0.95)
print(bf.summary(model, seq_len=4096))             # params / active params / FLOPs per block type
```

The model is a plain `nn.Module`, so FSDP2, torchtitan, Lightning and `torch.compile` work on it unchanged.

## Install

```bash
pip install brickformer                # core: torch + einops only
pip install "brickformer[fla]"         # fast Gated DeltaNet kernels (flash-linear-attention)
pip install "brickformer[mamba]"       # fast Mamba2 kernels (mamba-ssm; compiles against your torch/CUDA,
                                       # and this backend is not yet tested, see Limitations)
pip install "brickformer[yaml,hf]"     # YAML configs, HF Llama import
```

The example configs, the training script and the tests are in the
[GitHub repo](https://github.com/ahripon107/BrickFormer), not in the pip package. To use them, or
to develop, clone the repo and run `pip install -e ".[dev]"`.

## Components

| Slot | Registered types |
|---|---|
| Mixer (`MIXERS`) | `MHAttention`, `GQAttention`, `SlidingWindowAttention`, `MLA`, `Mamba2`, `GatedDeltaNet` |
| FFN (`FFNS`) | `SwiGLU`, `GeLUMLP`, `MoE` (top-k router, SwiGLU experts, shared experts, balance + z-loss) |
| Norm (`NORMS`) | `RMSNorm`, `LayerNorm`; attention takes `qk_norm=True` |
| Position (`POS_EMBEDS`) | `RoPE` (scaling: `linear`, `ntk`, `llama3`), `NoPE`; recurrent mixers need none |
| Residual | `pre_norm`, `post_norm`, `sandwich`, `parallel` |

A block is `mixer` + optional `ffn` (set it to `None` for pure Mamba blocks) + norms, wired by
`residual`. The model-level `norm`/`residual` are defaults that any block type can override.

To add your own component, register it and use it by name:

```python
@bf.MIXERS.register_module()
class MyMixer(bf.mixers.Mixer):
    def __init__(self, dim, ...): ...
    def init_state(self, batch_size, max_len, device=None, dtype=None): ...
    def forward(self, x, state=None, mask=None, position_ids=None):
        return y, state
```

## The mixer interface

```python
y, new_state = mixer(x, state=None, mask=None, position_ids=None)
state = mixer.init_state(batch_size, max_len, device, dtype)
```

Each mixer owns its state type: `KVCache` for attention, `MLACache` (compressed latent plus a shared
RoPE key, the memory saving MLA exists for), `Mamba2State` (conv + SSM state), and `GDNState`
(conv + `[K, V]` recurrent state). `CausalLM.init_cache()` holds one state per layer, so prefill,
chunked prefill and token-by-token decoding work for any hybrid stack.

## Patterns

`pattern` is a list of `block_types` keys or a compact string. In a string, a character that is
itself a key of `block_types` maps to that key. Otherwise these defaults apply:
`* / A → attn`, `W → swa`, `L → mla`, `M → mamba`, `G → gdn`. Override them with
`pattern_chars=dict(X='my_block')`. `-`, `_`, `|` and spaces are ignored. With `num_layers`, the
pattern is repeated to that depth.

## Backends

| Mixer | Default (`backend='auto'`) | Always available |
|---|---|---|
| Attention / MLA | `F.scaled_dot_product_attention` (uses FlashAttention kernels when it can) | yes |
| Gated DeltaNet | `fla` `chunk_gated_delta_rule` on CUDA when installed | chunked pure-PyTorch WY form |
| Mamba2 | `mamba_ssm` `mamba_chunk_scan_combined` on CUDA when installed | chunked pure-PyTorch SSD |

Setting `backend='fla'` or `'mamba_ssm'` without the package installed raises an `ImportError`
that names the package to install. Single-token decoding always uses a cheap recurrent update.
Recurrent scans accumulate in float32, or in float64 when the model runs in float64.

## Configs, checkpoints, HF

```python
model = bf.build_from_file('configs/jamba_like.py')   # .py (mmdet style), .yaml or .json
bf.dump_config(model.config, 'arch.yaml')              # round-trips exactly
bf.save_pretrained(model, 'ckpt/'); model = bf.load_pretrained('ckpt/')

from brickformer.hf import from_hf_llama
model = from_hf_llama('meta-llama/Llama-3.2-1B')       # needs transformers; logits match HF
```

Example configs in `configs/`: `llama_like.py`, `gdn_hybrid.py` (3:1 GDN/attention),
`jamba_like.py` (Mamba2 + attention + MoE), `moe.py` (MLA + shared-expert MoE) and
`swa_hybrid.yaml` (5:1 local/global attention, sandwich norm).

```
$ python -c "import brickformer as bf; print(bf.summary(bf.build_from_file('configs/moe.py'), 4096))"
block      layers        params        active   GFLOP/tok
dense           1    12,093,696    12,093,696      0.0347
moe            11   442,475,264   109,421,312      0.3341
embed                32,768,000
lm_head                       0                    0.0655
total          12   487,337,984                    0.4343   (seq_len=4096, forward)
```

FLOPs are forward estimates: 2 × active linear parameters per token, plus attention-score and
recurrent-state terms. Multiply by about 3 for a training step.

## Training example

```bash
python examples/train_tiny.py                       # char-level tiny-shakespeare, GGG* hybrid
python examples/train_tiny.py --pattern "MM*"       # Mamba2 + attention
python examples/train_tiny.py --config configs/moe.py --dim 256 --layers 8
```

On an RTX 5070 Ti, the default 3.2M-parameter GDN hybrid runs at about 355k tok/s with bf16
autocast. Val loss reaches about 1.51 at step 400 and then starts to overfit.

## Tests

```bash
pytest -q
```

- **Reference parity:** the Llama-style model matches HF `LlamaForCausalLM` logits in float64
  (GQA, MHA with tied embeddings, and llama3 RoPE scaling), and greedy generation matches token
  for token. The torch GDN kernel matches `fla`, and the chunked GDN and SSD kernels match their
  sequential recurrences.
- **Cache consistency:** for every mixer, a parallel forward equals token-by-token decoding and
  equals chunked prefill followed by decoding, at the mixer level and in a hybrid model.
- **Hybrid smoke test:** a model with every mixer plus MoE runs forward and backward (every
  parameter gets a gradient) and generates.
- **Config round-trip:** every example config is dumped to `.py`, `.yaml` and `.json` and rebuilt
  to an identical architecture; `save_pretrained`/`load_pretrained` reproduce the logits.
- Also covered: residual styles, left-padding masks, gradient checkpointing, and MoE routing
  (top-k = E equals the dense mixture).

## Limitations (v0.1)

- Sliding-window attention keeps a full-length KV cache and masks it, rather than using a ring buffer.
- `generate()` needs equal-length prompts. Key-padding masks are supported for training forwards only.
- MoE runs experts in a per-expert loop, with no grouped GEMM and no capacity dropping.
- MLA decompresses the latent cache at every step; there is no absorbed-matmul path.
- The `mamba_ssm` backend is wired up but untested in this repo, because it could not be built
  against torch 2.11 here. The pure-PyTorch Mamba2 path is fully tested.

## License

MIT. See [LICENSE](LICENSE).
