# RL stage (DPO): preference-tune an SFT checkpoint on UltraFeedback (chosen vs rejected replies).
#
#   brickformer-train configs/train/ultrafeedback_dpo.py --cfg-options init_from=runs/my_chat_sft
#
# The dataset stores chosen/rejected as full chat conversations; prompts are rendered with the
# tokenizer's chat template (ChatML-style if it has none). The reference model is a frozen copy
# of the starting policy unless objective.ref_model points at another checkpoint.

objective = dict(type="dpo", beta=0.1, loss_type="sigmoid")
init_from = "runs/gsm8k_sft"  # use a chat SFT checkpoint for real runs

_data = dict(path="trl-lib/ultrafeedback_binarized", max_len=1024, max_prompt_len=512)
data = dict(
    train=dict(_data, split="train"),  # type defaults to PreferenceDataset for objective='dpo'
    val=dict(_data, split="test", max_samples=500),
)

train = dict(
    type="HFTrainer",
    output_dir="runs/ultrafeedback_dpo",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # pairs; each step runs 8 sequences through policy and reference
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,
    gradient_checkpointing=False,
    logging_steps=10,
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
)
