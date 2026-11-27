# Pretraining with a data-mixing schedule: the Qwen3-0.6B architecture from scratch on web text, math
# and code, moving to more math and code for the last 20% of training ("annealing").
#
#   brickformer-train configs/train/qwen3_0p6b_mixture.py
#   brickformer-train configs/train/qwen3_0p6b_mixture.py --cfg-options train.max_steps=5000
#
# Schedule positions (`until`) are fractions of train.max_steps, so the phases scale with the run;
# use `until_tokens=...` for fixed token budgets, or `at=` keypoints for linear blending. Every
# 1024-token block comes from one source, and the trainer logs each source's share as mix/<name>.
# Validation is per source: eval_web_loss, eval_math_loss, eval_code_loss.

model = dict(
    type="CausalLM",
    vocab_size=151936,
    dim=1024,
    pattern="A",
    num_layers=28,
    block_types=dict(
        attn=dict(
            mixer=dict(type="GQAttention", heads=16, kv_heads=8, head_dim=128, qk_norm=True,
                       rope=dict(type="RoPE", theta=1_000_000.0)),
            ffn=dict(type="SwiGLU", hidden=3072),
        ),
    ),
    norm=dict(type="RMSNorm", eps=1e-6),
    tie_embeddings=True,
    grad_checkpointing=True,
)

tokenizer = dict(type="HFTokenizer", name="Qwen/Qwen3-0.6B")

# held-out validation files: fineweb-edu shard 013 (as in qwen3_0p6b.py), finemath shard 63;
# codeparrot-clean ships a separate validation set
_web = "sample/10BT/{:03d}_00000.parquet"
_math = "finemath-4plus/train-{:05d}-of-00064.parquet"
_sources = dict(
    web=dict(path="HuggingFaceFW/fineweb-edu", data_files=[_web.format(i) for i in range(13)],
             streaming=True, shuffle_buffer=10_000),
    math=dict(path="HuggingFaceTB/finemath", data_files=[_math.format(i) for i in range(63)], streaming=True,
              shuffle_buffer=10_000),
    code=dict(path="codeparrot/codeparrot-clean", text_field="content", streaming=True, shuffle_buffer=10_000),
)

data = dict(
    train=dict(
        type="MixturePretrainDataset",
        seq_len=1024,
        sources=_sources,
        schedule=[
            dict(until=0.8, weights=dict(web=0.85, math=0.10, code=0.05)),
            dict(until=1.0, weights=dict(web=0.40, math=0.35, code=0.25)),
        ],
    ),
    val=dict(  # one validation set per source -> eval_<name>_loss
        web=dict(path="HuggingFaceFW/fineweb-edu", data_files=[_web.format(13)], streaming=True,
                 max_samples=300, seq_len=1024),
        math=dict(path="HuggingFaceTB/finemath", data_files=[_math.format(63)], streaming=True, max_samples=300,
                  seq_len=1024),
        code=dict(path="codeparrot/codeparrot-clean-valid", text_field="content", streaming=True,
                  max_samples=300, seq_len=1024),
    ),
)

# zero-shot benchmarks every eval_steps and at the end: bench/<task>_<metric>, bench/avg, and
# runs/qwen3_0p6b_mixture/benchmarks.jsonl. A fixed 500-example subset per task keeps a run to about a
# minute; score the final checkpoint on full splits with `brickformer-eval runs/qwen3_0p6b_mixture`.
# batch_tokens=2048: eval logits (2048 x 152k vocab, bf16) take ~0.6 GB next to the ~12.5 GB training peak.
benchmarks = dict(
    tasks=["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "lambada"],
    max_samples=500,
    batch_tokens=2048,
)

train = dict(
    type="HFTrainer",
    output_dir="runs/qwen3_0p6b_mixture",
    max_steps=1_500,
    per_device_train_batch_size=1,  # x32 accumulation = 32k tokens per step; peak ~12 GiB on 16 GB
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=6e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.1,
    adam_beta2=0.95,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",
    bf16=True,
    logging_steps=10,
    eval_steps=250,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    seed=42,
)
