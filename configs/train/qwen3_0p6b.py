# Experiment config: the Qwen3-0.6B architecture (~596M params), pretrained from scratch on
# FineWeb-Edu with the Qwen3 tokenizer.
#
#   brickformer-train configs/train/qwen3_0p6b.py
#   brickformer-train configs/train/qwen3_0p6b.py --cfg-options train.max_steps=5000
#   brickformer-train configs/train/qwen3_0p6b.py --resume
#
# Architecture follows Qwen/Qwen3-0.6B's config.json: 28 layers, hidden 1024, 16 query / 8 KV heads
# with head_dim 128 (so q_proj is 1024 -> 2048), QK-norm, RoPE theta 1e6, SwiGLU 3072, RMSNorm
# eps 1e-6, tied embeddings, vocab 151936. Weights are randomly initialised, not imported.
#
# Memory (single 16 GB GPU): fp32 weights + AdamW states are ~9.5 GB, so activations are kept small
# with gradient checkpointing and micro-batches of 1 x 1024 tokens (peak ~12.2 GiB).

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
    residual="pre_norm",
    tie_embeddings=True,
    grad_checkpointing=True,
)

tokenizer = dict(type="HFTokenizer", name="Qwen/Qwen3-0.6B")

# fineweb-edu has only a train split: shards 000-012 of sample-10BT train, shard 013 is held out
_shards = "sample/10BT/{:03d}_00000.parquet"
_data = dict(type="PretrainDataset", path="HuggingFaceFW/fineweb-edu", split="train", text_field="text",
             seq_len=1024, streaming=True)
data = dict(
    train=dict(_data, data_files=[_shards.format(i) for i in range(13)], shuffle_buffer=10_000),
    val=dict(_data, data_files=[_shards.format(13)], max_samples=500),
)

train = dict(
    type="HFTrainer",
    output_dir="runs/qwen3_0p6b",
    max_steps=1_000,
    # x32 accumulation = 32 x 1024 = 32k tokens per step. Measured peak on a 16 GB GPU: micro-batch 1
    # = 12.2 GiB reserved, 2 = 14.3 GiB (too close to the limit), 4 = OOM (the fp32 logits for the
    # 152k vocab alone are 2.3 GB at 4 x 1024).
    per_device_train_batch_size=1,
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
    dataloader_num_workers=0,
    report_to="none",
    seed=42,
)
