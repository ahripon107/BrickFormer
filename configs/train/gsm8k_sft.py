# SFT stage: fine-tune a pretrained BrickFormer checkpoint on GSM8K worked solutions.
# The loss covers the answer tokens only; the "Question: ... Answer:" prompt is masked.
#
#   brickformer-train configs/train/gsm8k_sft.py
#   brickformer-train configs/train/gsm8k_sft.py --cfg-options init_from=runs/my_pretrain
#
# model and tokenizer come from the init_from checkpoint (its config.json / experiment.json).

objective = "sft"
init_from = "runs/tinystories_gdn"

_data = dict(path="openai/gsm8k", name="main", prompt_template="Question: {question}\nAnswer: ",
             completion_field="answer", max_len=512)
data = dict(
    train=dict(_data, split="train"),  # type defaults to SFTDataset for objective='sft'
    val=dict(_data, split="test", max_samples=500),
)

train = dict(
    type="HFTrainer",
    output_dir="runs/gsm8k_sft",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    weight_decay=0.0,
    bf16=True,
    logging_steps=20,
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
)
