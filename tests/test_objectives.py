"""Objective-specific data loading and losses (SFT, DPO, GRPO), offline with the byte tokenizer."""

import json
import math

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")

import brickformer as bf  # noqa: E402
from brickformer.data import (  # noqa: E402
    ByteTokenizer,
    PreferenceCollator,
    PreferenceDataset,
    PromptCollator,
    PromptDataset,
    SFTDataset,
    split_prompt_response,
    tokenize_conversation,
)
from brickformer.train import build_experiment, run_experiment, token_logps  # noqa: E402
from brickformer.train.rewards import NumberMatch, RegexMatch  # noqa: E402

TOK = ByteTokenizer()
EOS = TOK.eos_token_id


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def response_text(ids, labels):
    return TOK.decode([i for i, lab in zip(ids, labels) if lab != -100])


def tiny_model_cfg():
    ffn = dict(type="SwiGLU", hidden=128)
    return dict(type="CausalLM", vocab_size="auto", dim=64, pattern="G*", tie_embeddings=True,
                block_types=dict(attn=dict(mixer=dict(type="GQAttention", heads=4), ffn=ffn),
                                 gdn=dict(mixer=dict(type="GatedDeltaNet", heads=2, backend="torch"), ffn=ffn)))


def train_cfg(tmp_path, name, **kw):
    cfg = dict(type="HFTrainer", output_dir=str(tmp_path / name), max_steps=4, per_device_train_batch_size=4,
               learning_rate=3e-3, warmup_steps=0, logging_steps=2, save_steps=1000, use_cpu=True,
               disable_tqdm=True, seed=0)
    cfg.update(kw)
    return cfg


# ------------------------------------------------------------------------------- chat formatting
def test_conversation_masks_only_assistant_turns():
    msgs = [dict(role="system", content="Be brief."), dict(role="user", content="Hi"),
            dict(role="assistant", content="Hello!"), dict(role="user", content="2+2?"),
            dict(role="assistant", content="4")]
    ids, labels = tokenize_conversation(TOK, msgs)
    assert TOK.decode(ids) == TOK.render_chat(msgs)  # trained text is exactly the rendered chat
    trained = [i for i, lab in zip(ids, labels) if lab != -100]
    assert trained.count(EOS) == 2  # each reply ends with EOS
    assert TOK.decode(trained) == "Hello!<|im_end|>\n4<|im_end|>\n"
    with pytest.raises(ValueError, match="no assistant"):
        tokenize_conversation(TOK, msgs[:2])


@pytest.mark.parametrize("prompt,response,want_prompt,want_resp", [
    ("Q: 1+1? A:", " 2", "Q: 1+1? A:", " 2"),  # plain text
    ("1+1?", [dict(role="assistant", content="2")],
     "<|im_start|>user\n1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>\n"),
    (None, [dict(role="user", content="1+1?"), dict(role="assistant", content="2")],  # full conversation
     "<|im_start|>user\n1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>\n"),
    ("1+1?", [dict(role="user", content="1+1?"), dict(role="assistant", content="2")],  # ultrafeedback-style
     "<|im_start|>user\n1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>\n"),
])
def test_prompt_response_formats(prompt, response, want_prompt, want_resp):
    p, r = split_prompt_response(TOK, prompt, response)
    assert TOK.decode(p) == want_prompt and TOK.decode(r) == want_resp and r[-1] == EOS


def test_non_prefix_stable_template_is_rejected():
    class Bad(ByteTokenizer):
        def render_chat(self, messages, add_generation_prompt=False):
            return "".join(m["content"] for m in reversed(messages)) + ("|" if add_generation_prompt else "")

    with pytest.raises(ValueError, match="prefix-stable"):
        split_prompt_response(Bad(), "q", [dict(role="assistant", content="a")])


# ------------------------------------------------------------------------------- datasets per objective
def test_sft_dataset_formats(tmp_path):
    qa = write_jsonl(tmp_path / "qa.jsonl", [dict(question=f"{i}+1", answer=str(i + 1)) for i in range(5)])
    ds = SFTDataset("json", TOK, data_files=qa, prompt_template="Q: {question}\nA: ", completion_field="answer")
    assert len(ds) == 5
    assert TOK.decode(ds[3]["input_ids"]) == "Q: 3+1\nA: 4"
    assert response_text(ds[3]["input_ids"], ds[3]["labels"]) == "4"

    chat = write_jsonl(tmp_path / "chat.jsonl", [dict(messages=[dict(role="user", content="hi"),
                                                                 dict(role="assistant", content="yo")])] * 3)
    ds = SFTDataset("json", TOK, data_files=chat)
    assert response_text(ds[0]["input_ids"], ds[0]["labels"]) == "yo<|im_end|>\n"

    # prompts that fill max_len leave no response tokens -> dropped
    rows = [dict(prompt="x" * 50, completion="y"), dict(prompt="a", completion="b")]
    long = write_jsonl(tmp_path / "long.jsonl", rows)
    ds = SFTDataset("json", TOK, data_files=long, max_len=20)
    assert len(ds) == 1


def test_preference_dataset_and_collator(tmp_path):
    rows = [dict(prompt=f"Is {i} even?", chosen=" yes" if i % 2 == 0 else " no",
                 rejected=" no" if i % 2 == 0 else " yes") for i in range(4)]
    ds = PreferenceDataset("json", TOK, data_files=write_jsonl(tmp_path / "pref.jsonl", rows))
    batch = PreferenceCollator(pad_token_id=0)([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 4  # chosen x2 then rejected x2
    assert response_text(batch["input_ids"][0].tolist(), batch["labels"][0].tolist()) == " yes"
    assert response_text(batch["input_ids"][2].tolist(), batch["labels"][2].tolist()) == " no"


def test_prompt_dataset_keeps_reward_columns(tmp_path):
    rows = [dict(question="2+2?", answer="#### 4"), dict(question="x" * 100, answer="#### 0")]
    ds = PromptDataset("json", TOK, data_files=write_jsonl(tmp_path / "p.jsonl", rows), prompt_field="question",
                       max_prompt_len=50)
    assert len(ds) == 1  # over-long prompt dropped, not truncated
    batch = PromptCollator()([ds[0]])
    assert TOK.decode(batch["prompt_ids"][0]) == "2+2?" and batch["columns"]["answer"] == ["#### 4"]


def test_objective_picks_and_validates_dataset_type(tmp_path):
    data_file = write_jsonl(tmp_path / "qa.jsonl", [dict(prompt="a", completion="b")])
    base = dict(model=tiny_model_cfg(), tokenizer=dict(type="ByteTokenizer"), train=train_cfg(tmp_path, "x"))
    exp = build_experiment(dict(base, objective="sft", data=dict(train=dict(path="json", data_files=data_file))))
    assert exp.objective.name == "sft" and "labels" in exp.train_dataset.column_names
    with pytest.raises(ValueError, match="is for objective"):
        build_experiment(dict(base, objective="sft", data=dict(train=dict(type="PreferenceDataset", path="json"))),
                         build_data=False)
    with pytest.raises(ValueError, match="at least one reward"):
        build_experiment(dict(base, objective=dict(type="grpo", rewards=[]), data=dict(train=dict(path="json"))),
                         build_data=False)


# ------------------------------------------------------------------------------- losses
def test_token_logps_matches_log_softmax():
    logits = torch.randn(2, 5, 11)
    labels = torch.tensor([[1, 2, 3, -100, -100], [-100, 4, 5, 6, 7]])
    logp, mask = token_logps(logits, labels)
    ref = torch.log_softmax(logits[:, :-1], -1).gather(-1, labels[:, 1:].clamp(min=0)[..., None]).squeeze(-1)
    torch.testing.assert_close(logp, ref * mask)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]


def test_sft_loss_covers_responses_only(tmp_path):
    model = bf.build_model(dict(tiny_model_cfg(), vocab_size=320))
    ids = torch.randint(0, 256, (1, 10))
    labels = ids.clone()
    labels[:, :6] = -100
    loss = model(ids, labels=labels).loss
    logp, mask = token_logps(model(ids).logits, labels)
    torch.testing.assert_close(loss, -(logp.sum() / mask.sum()))


def test_rewards():
    prompts = ["q"] * 4
    nm = NumberMatch(reference_field="answer")
    assert nm(prompts, ["so 4", "it is 1,000", "x", "4.0"], answer=["#### 4", "#### 1000", "#### 4", "4"]) == \
        [1.0, 1.0, 0.0, 1.0]
    assert RegexMatch(r"####")(prompts, ["#### 4", "4"] * 2) == [1.0, 0.0, 1.0, 0.0]


# ------------------------------------------------------------------------------- end to end
def test_sft_then_dpo_chain(tmp_path):
    """SFT a model, then DPO from that checkpoint (model + tokenizer come from init_from)."""
    rows = [dict(prompt=f"Is {i} even?", chosen=" yes" if i % 2 == 0 else " no",
                 rejected=" no" if i % 2 == 0 else " yes") for i in range(16)]
    pref = write_jsonl(tmp_path / "pref.jsonl", rows)
    sft_rows = write_jsonl(tmp_path / "sft.jsonl", [dict(prompt=r["prompt"], completion=r["chosen"]) for r in rows])

    sft = run_experiment(dict(objective="sft", model=tiny_model_cfg(), tokenizer=dict(type="ByteTokenizer"),
                              data=dict(train=dict(path="json", data_files=sft_rows)),
                              train=train_cfg(tmp_path, "sft")))
    assert all(math.isfinite(h["loss"]) for h in sft.state.log_history if "loss" in h)

    dpo_cfg = dict(objective=dict(type="dpo", beta=0.5), init_from=str(tmp_path / "sft"),
                   data=dict(train=dict(path="json", data_files=pref), val=dict(path="json", data_files=pref)),
                   train=train_cfg(tmp_path, "dpo", max_steps=20, learning_rate=1e-3, eval_steps=20))
    exp = build_experiment(dpo_cfg)
    # starts from the SFT weights, and at step 0 policy == reference so the DPO loss is log(2)
    sft_model = bf.load_pretrained(tmp_path / "sft")
    for a, b in zip(exp.model.state_dict().values(), sft_model.state_dict().values()):
        assert torch.equal(a, b)

    class Stub:  # minimal trainer interface used by objectives
        ref_model = exp.objective.build_ref_model(exp.model)

        def record(self, name, value):
            pass

    batch = exp.objective.collator(exp.tokenizer)([exp.train_dataset[i] for i in range(4)])
    with torch.no_grad():
        assert abs(exp.objective.compute_loss(Stub(), exp.model, batch).item() - math.log(2)) < 1e-5

    trainer = exp.run()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    assert "rewards/accuracy" in logs[0] and "rewards/margin" in logs[0]
    final = [h for h in trainer.state.log_history if "eval_loss" in h][-1]  # all 16 pairs
    print(final)
    assert final["eval_loss"] < math.log(2) - 0.1
    assert final["eval_rewards/accuracy"] >= 0.75 and final["eval_rewards/margin"] > 0
    # the reference stayed frozen at the SFT weights
    for a, b in zip(trainer.ref_model.state_dict().values(), sft_model.state_dict().values()):
        torch.testing.assert_close(a.cpu(), b)


def ascii_fraction(prompts, completions, **columns):
    """Toy reward used by the GRPO test: share of completion characters that are lowercase a-m."""
    return [sum("a" <= ch <= "m" for ch in c) / max(len(c), 1) for c in completions]


def test_grpo_learns_toy_reward(tmp_path):
    rows = [dict(prompt=f"Write {i}: ", target=str(i)) for i in range(32)]
    data = write_jsonl(tmp_path / "prompts.jsonl", rows)
    cfg = dict(objective=dict(type="grpo", num_generations=8, max_completion_len=12, temperature=1.0, beta=0.02,
                              rewards=[dict(type="PythonReward", fn="test_objectives:ascii_fraction"),
                                       dict(type="RegexMatch", pattern="q", weight=0.0)]),
               model=tiny_model_cfg(), tokenizer=dict(type="ByteTokenizer"),
               data=dict(train=dict(path="json", data_files=data),
                         val=dict(path="json", data_files=data, max_samples=4)),
               train=train_cfg(tmp_path, "grpo", max_steps=30, learning_rate=1e-2, logging_steps=5, eval_steps=30))
    trainer = run_experiment(cfg)
    logs = [h for h in trainer.state.log_history if "loss" in h]
    for key in ("reward", "reward_std", "rewards/ascii_fraction", "rewards/RegexMatch", "kl", "completion_length",
                "frac_truncated"):
        assert key in logs[0], key
    assert logs[-1]["reward"] > logs[0]["reward"] + 0.1, [h["reward"] for h in logs]
    assert any("eval_reward" in h for h in trainer.state.log_history)
    assert (tmp_path / "grpo" / "model.pt").is_file()
