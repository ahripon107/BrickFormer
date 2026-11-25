# SFT from pretrained HF weights: fine-tune Qwen3-0.6B-Base on GSM8K worked solutions.
#
#   brickformer-train configs/train/qwen3_0p6b_gsm8k_sft.py --dry-run   # downloads, checks, loads the weights
#   brickformer-train configs/train/qwen3_0p6b_gsm8k_sft.py
#
# `pretrained` takes a Hub id or a local HF directory (Llama, Mistral, Qwen2, Qwen3). Without a full
# `model` section the architecture comes from the checkpoint's config.json; the partial section below
# only turns on gradient checkpointing. To use your own model config instead, give a full one (e.g.
# the `model` of qwen3_0p6b.py): the weights are loaded only if it matches the checkpoint exactly,
# otherwise every difference is listed. The tokenizer defaults to the checkpoint's own.

objective = "sft"
pretrained = "Qwen/Qwen3-0.6B-Base"  # or dict(path=..., revision=..., strict=True)
model = dict(grad_checkpointing=True)

_data = dict(path="openai/gsm8k", name="main", prompt_template="Question: {question}\nAnswer: ",
             completion_field="answer", max_len=512)
data = dict(
    train=dict(_data, split="train"),
    val=dict(_data, split="test", max_samples=500),
)

train = dict(
    type="HFTrainer",
    output_dir="runs/qwen3_0p6b_gsm8k_sft",
    num_train_epochs=2,
    per_device_train_batch_size=2,  # x8 accumulation = 16 sequences per step; fits 16 GB
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=30,
    weight_decay=0.0,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",
    bf16=True,
    logging_steps=10,
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)
