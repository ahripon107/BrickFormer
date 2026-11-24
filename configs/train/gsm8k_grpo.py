# RL stage (GRPO): sample completions for GSM8K questions and reward correct final answers.
#
#   brickformer-train configs/train/gsm8k_grpo.py
#
# Each step samples num_generations completions for each of per_device_train_batch_size prompts
# (here 4 x 8 = 32 completions), scores them with the rewards below, and pushes the policy towards
# the better-than-group-average ones. Start from an SFT checkpoint so some answers are right.

objective = dict(
    type="grpo",
    num_generations=8,
    max_completion_len=256,
    temperature=1.0,
    top_p=1.0,
    beta=0.0,  # > 0 adds a KL penalty to a frozen copy of the starting policy
    scale_rewards=True,
    loss_agg="token",
    rewards=[
        dict(type="NumberMatch", reference_field="answer"),  # final number matches the "#### 42" answer
        dict(type="RegexMatch", pattern=r"####\s*-?[\d,.]+", weight=0.1),  # uses the answer format
        # dict(type="PythonReward", fn="my_package.rewards:my_reward"),  # any custom function
    ],
)
init_from = "runs/gsm8k_sft"

_data = dict(path="openai/gsm8k", name="main", prompt_template="Question: {question}\nAnswer: ",
             max_prompt_len=256)
data = dict(
    train=dict(_data, split="train"),  # type defaults to PromptDataset; the 'answer' column feeds the rewards
    val=dict(_data, split="test", max_samples=64),
)

train = dict(
    type="HFTrainer",
    output_dir="runs/gsm8k_grpo",
    max_steps=500,
    per_device_train_batch_size=4,  # prompts per step
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=5,
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
)
