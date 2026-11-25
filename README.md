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
pip install "brickformer[yaml,hf]"     # YAML configs, HF Llama / Mistral / Qwen import
pip install "brickformer[train]"       # config-driven training (HF Trainer + datasets)
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

from brickformer.hf import from_hf, load_hf_weights
model = from_hf('Qwen/Qwen3-0.6B')                     # Llama / Mistral / Qwen2 / Qwen3; logits match HF
model = load_hf_weights(bf.build_from_file('my_qwen3.py'), 'Qwen/Qwen3-0.6B')  # your config, checked first
print(bf.architecture_diff(my_cfg, other_cfg))         # what differs between two configs or models
```

`load_hf_weights` loads a checkpoint into a model built from your own config only if the config
describes exactly the checkpoint's architecture: module types, heads, `head_dim`, QK-norm, biases,
RoPE theta and scaling, norm eps, sliding windows, tying and every parameter shape. Otherwise it
raises `ArchitectureMismatch` and lists each difference, grouped across layers:

```
model config does not match the architecture of 'Qwen/Qwen3-0.6B' (8 differences):
  blocks.[0-27].mixer.head_dim: config=64, pretrained=128
  blocks.[0-27].mixer.q_proj.weight: parameter shape config=[1024, 1024], pretrained=[2048, 1024]
  ...
```

`strict=False` only requires parameter names and shapes to match, so you can deliberately change
settings that carry no weights, such as RoPE scaling to extend the context. Weights are read straight
from the `*.safetensors` files, without instantiating the HF model. Sliding-window layers (Mistral, and
Qwen2/Qwen3 `layer_types`) become `SlidingWindowAttention` blocks.

Example configs in `configs/`: `llama_like.py`, `gdn_hybrid.py` (3:1 GDN/attention),
`jamba_like.py` (Mamba2 + attention + MoE), `moe.py` (MLA + shared-expert MoE) and
`swa_hybrid.yaml` (5:1 local/global attention, sandwich norm). Full experiment configs
(model + data + training) are in `configs/train/`.

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

## Training from config

One experiment file describes the objective, model, tokenizer, data and training run. The HF
`Trainer` runs it (install `brickformer[train]`):

```python
objective = 'pretrain'                      # 'sft' | dict(type='dpo', ...) | dict(type='grpo', ...)
model = dict(type='CausalLM', vocab_size='auto', dim=512, pattern='GGG*', num_layers=8, ...)
tokenizer = dict(type='HFTokenizer', name='gpt2')                  # or dict(type='ByteTokenizer')
data = dict(
    train=dict(path='roneneldan/TinyStories', split='train', seq_len=512,
               streaming=True, shuffle_buffer=10_000),
    val=dict(path='roneneldan/TinyStories', split='validation', seq_len=512, max_samples=2_000),
)
train = dict(type='HFTrainer', output_dir='runs/tinystories_gdn', max_steps=3_000,
             per_device_train_batch_size=16, gradient_accumulation_steps=2, learning_rate=1e-3,
             lr_scheduler_type='cosine', warmup_steps=200, bf16=True, eval_steps=500, save_steps=1_000)
```

```bash
brickformer-train configs/train/tinystories_gdn.py
brickformer-train configs/train/tinystories_gdn.py --cfg-options train.learning_rate=3e-4 model.dim=768
brickformer-train configs/train/tinystories_gdn.py --dry-run     # print resolved config + param/FLOP summary
brickformer-train configs/train/tinystories_gdn.py --resume      # continue from the latest checkpoint
```

### Objectives

The `objective` decides how the data is loaded and formatted, how batches are collated, and the
loss. Each data entry defaults to the objective's dataset type, so you usually omit `type`; a
dataset type built for another objective is rejected.

| `objective` | Dataset type (default) | Data it reads | Loss |
|---|---|---|---|
| `'pretrain'` | `PretrainDataset` | a text column | next-token loss on packed `seq_len` blocks |
| `'sft'` | `SFTDataset` | chat `messages`, or prompt + completion | next-token loss on response tokens only |
| `dict(type='dpo', beta=0.1)` | `PreferenceDataset` | prompt, chosen, rejected | DPO against a frozen reference model |
| `dict(type='grpo', rewards=[...])` | `PromptDataset` | prompts, plus columns for the rewards | GRPO on completions sampled during training |

- **Prompts:** `prompt_template='Question: {question}\nAnswer: '` builds a prompt from any columns
  of a row. Otherwise `prompt_field` and `completion_field` name the columns.
- **Chat data:** message lists (`[{'role': ..., 'content': ...}]`) are rendered with the
  tokenizer's chat template, or a ChatML-style one if it has none, such as GPT-2. Multi-turn SFT
  trains on every assistant turn. DPO accepts the common TRL dataset layouts, for example
  `trl-lib/ultrafeedback_binarized`. Every response ends with EOS, so the model learns to stop.
- **DPO:** `loss_type='sigmoid'` or `'ipo'`, plus `label_smoothing`. The reference model is a
  frozen copy of the starting policy, or another checkpoint via `ref_model='path'`. Logs
  `rewards/chosen`, `rewards/rejected`, `rewards/accuracy` and `rewards/margin`.
- **GRPO:** for each prompt it samples `num_generations` completions with BrickFormer's
  `generate()` and scores them with the weighted `rewards`. Built-in rewards are `NumberMatch`
  (GSM8K-style final answers), `ExactMatch`, `RegexMatch`, `LengthPenalty`, and `PythonReward`
  (`fn='pkg.module:function'`). Advantages are normalized within each group (`scale_rewards=False`
  gives Dr. GRPO). `beta > 0` adds a KL penalty to a frozen reference model. Logs `reward`,
  `reward_std`, `frac_reward_zero_std`, each reward, `completion_length` and `frac_truncated`.
  `per_device_train_batch_size` counts prompts.
- **Chaining stages:** `init_from='runs/sft'` starts from a checkpoint. The model and tokenizer
  come from that checkpoint unless you override them, so stages chain:
  `tinystories_gdn.py` → `gsm8k_sft.py` → `gsm8k_grpo.py` or `ultrafeedback_dpo.py`, all in
  `configs/train/`.
- **Starting from HF weights:** `pretrained='Qwen/Qwen3-0.6B-Base'` (a Hub id or local HF directory,
  or `dict(path=..., revision=..., strict=True)`) loads Llama, Mistral, Qwen2 or Qwen3 weights. With a
  full `model` section, training starts only if that config matches the checkpoint's architecture
  (see `load_hf_weights` above). Without one, the architecture comes from the checkpoint's
  `config.json`, and a partial section such as `model = dict(grad_checkpointing=True)` adds settings
  on top. `vocab_size='auto'` takes the checkpoint's vocabulary and the tokenizer defaults to the
  checkpoint's own. `--dry-run` downloads the weights and runs the check without training. Example:
  `configs/train/qwen3_0p6b_gsm8k_sft.py`. `configs/train/qwen3_0p6b.py` is the same architecture
  trained from scratch.

### More details

- **`model.vocab_size='auto'`:** uses the tokenizer's vocabulary rounded up to a multiple of 64
  (GPT-2: 50257 → 50304), for faster matmuls.
- **Pretraining data:** each document gets an EOS token, then documents are packed into full
  `seq_len` blocks, so no compute goes to padding (`packing=False` truncates and pads instead).
  `streaming=True` tokenizes on the fly and needs `max_steps`. All dataset types read Hub datasets
  and local files (`path='json'`, `'parquet'` or `'text'` with `data_files=...`).
- **`train`:** takes any `transformers.TrainingArguments` field. Evaluation always runs the
  objective's loss, never gathers vocab-sized logits, and also reports the objective's metrics
  with an `eval_` prefix. No external logger is used unless you set `report_to`.
- **Checkpoints:** saved in BrickFormer format with the resolved `experiment.json` and the
  tokenizer. Any checkpoint loads with `bf.load_pretrained(dir)`, and
  `brickformer-train dir/experiment.json` reruns the experiment.
- **MoE:** the aux loss is part of the training loss and is also logged separately as `aux_loss`.

`train` picks its backend by `type`, so you can add another (a custom loop, or torchtitan for
FSDP2) by registering it in `bf.RUNNERS`.

On an RTX 5070 Ti, `configs/train/tinystories_gdn.py` (52.6M params) trains at about 51k tok/s
on streamed data. Val loss reaches 2.52 after 400 steps (about 2 minutes), and the model already
writes coherent TinyStories text. Continuing from that checkpoint with `gsm8k_sft.py` for 300
steps (21 s) lowers the GSM8K answer-token eval loss from 3.33 to 2.99. Then `gsm8k_grpo.py` runs
at about 3 s per step for 32 sampled completions; the model uses the `####` answer format 70–80% of
the time, but a 52M-param story model almost never gets the number right.
`ultrafeedback_dpo.py` also runs from the SFT checkpoint. On real tasks these are integration
checks, not quality results.

### Minimal training script

`examples/train_tiny.py` is a dependency-free loop, about 150 lines, on character-level
tiny-shakespeare:

```bash
python examples/train_tiny.py                       # GGG* hybrid
python examples/train_tiny.py --pattern "MM*"       # Mamba2 + attention
python examples/train_tiny.py --config configs/moe.py --dim 256 --layers 8
```

The default 3.2M-parameter GDN hybrid runs at about 355k tok/s with bf16 autocast. Val loss
reaches about 1.51 at step 400 and then starts to overfit.

## Tests

```bash
pytest -q
```

- **Reference parity:** the Llama-style model matches HF `LlamaForCausalLM` logits in float64
  (GQA, MHA with tied embeddings, and llama3 RoPE scaling), and greedy generation matches token
  for token. Imported Qwen3 (QK-norm, `head_dim` ≠ `dim / heads`, tied), Qwen2 (q/k/v bias, mixed
  sliding-window layers) and Mistral (sliding window) models match HF logits in float64 too, and the
  real `Qwen/Qwen3-0.6B` matches exactly. The torch GDN kernel matches `fla`, and the chunked GDN and
  SSD kernels match their sequential recurrences.
- **Architecture checks:** equivalent configs written differently compare equal, every kind of
  difference is reported, mismatched configs are refused in `load_hf_weights` and in experiments,
  and `strict=False` still refuses wrong parameter shapes.
- **Cache consistency:** for every mixer, a parallel forward equals token-by-token decoding and
  equals chunked prefill followed by decoding, at the mixer level and in a hybrid model.
- **Hybrid smoke test:** a model with every mixer plus MoE runs forward and backward (every
  parameter gets a gradient) and generates.
- **Config round-trip:** every example config is dumped to `.py`, `.yaml` and `.json` and rebuilt
  to an identical architecture; `save_pretrained`/`load_pretrained` reproduce the logits.
- Also covered: residual styles, left-padding masks, gradient checkpointing, and MoE routing
  (top-k = E equals the dense mixture).
- **Objectives (offline):** chat rendering and masking (multi-turn, the common prompt/response
  layouts, rejection of chat templates whose renderings aren't prefixes of each other), each
  objective's dataset type and how mismatches are rejected, SFT loss on response tokens only, the
  reward functions, an SFT → DPO chain via `init_from`, and GRPO. DPO starts at exactly log 2 and
  reaches 100% preference accuracy; GRPO raises a toy reward.
- **Training (offline, byte tokenizer + local text):** packing and collation, streaming versus
  map-style data, `vocab_size='auto'`, `--cfg-options` overrides, and an end-to-end HF Trainer
  run with eval, MoE aux-loss logging and checkpoints (tied embeddings included). Checkpoints
  reload to identical logits, a resumed run continues from step 3 rather than restarting, bf16
  with gradient accumulation works (CUDA), and the CLI runs both `--dry-run` and real training.

## Limitations (v0.1)

- Sliding-window attention keeps a full-length KV cache and masks it, rather than using a ring buffer.
- `generate()` needs equal-length prompts. Key-padding masks are supported for training forwards only.
- MoE runs experts in a per-expert loop, with no grouped GEMM and no capacity dropping.
- MLA decompresses the latent cache at every step; there is no absorbed-matmul path.
- The `mamba_ssm` backend is wired up but untested in this repo, because it could not be built
  against torch 2.11 here. The pure-PyTorch Mamba2 path is fully tested.

## License

MIT. See [LICENSE](LICENSE).
