# Experiment config: a ~50M-param GDN/attention hybrid on TinyStories, trained with the HF Trainer.
#
#   brickformer-train configs/train/tinystories_gdn.py
#   brickformer-train configs/train/tinystories_gdn.py --cfg-options train.max_steps=500 model.dim=256
#   brickformer-train configs/train/tinystories_gdn.py --resume          # continue the latest checkpoint
#
# Needs `pip install brickformer[train]`. Data is streamed, so nothing large is downloaded up front.

_ffn = dict(type="SwiGLU", hidden=1408)

model = dict(
    type="CausalLM",
    vocab_size="auto",  # tokenizer vocab rounded up to a multiple of 64 (50257 -> 50304)
    dim=512,
    pattern="GGG*",
    num_layers=8,
    block_types=dict(
        attn=dict(mixer=dict(type="GQAttention", heads=8, kv_heads=4, qk_norm=True), ffn=_ffn),
        gdn=dict(mixer=dict(type="GatedDeltaNet", heads=8, head_dim=64), ffn=_ffn),
    ),
    norm=dict(type="RMSNorm"),
    tie_embeddings=True,
)

tokenizer = dict(type="HFTokenizer", name="gpt2")

_data = dict(type="HFTextDataset", path="roneneldan/TinyStories", text_field="text", seq_len=512)
data = dict(
    train=dict(_data, split="train", streaming=True, shuffle_buffer=10_000),
    val=dict(_data, split="validation", max_samples=2_000),
)

train = dict(
    type="HFTrainer",
    output_dir="runs/tinystories_gdn",
    max_steps=3_000,
    per_device_train_batch_size=16,  # x2 accumulation = 32 x 512 tokens per step; fits 16 GB
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-3,
    lr_scheduler_type="cosine",
    warmup_steps=200,
    weight_decay=0.1,
    adam_beta2=0.95,
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=50,
    eval_steps=500,
    save_steps=1_000,
    save_total_limit=2,
    report_to="none",  # "wandb" / "tensorboard"
    seed=42,
)
