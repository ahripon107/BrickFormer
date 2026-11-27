"""Benchmark evaluation: prompt formats, scoring, metrics and the training integration (offline)."""

import json
import math

import pytest
import torch

import brickformer as bf
from brickformer.data import ByteTokenizer
from brickformer.evaluation import Benchmark, BenchmarkSuite, Example, evaluate_benchmark, score_requests
from brickformer.evaluation import benchmarks as bm
from brickformer.registry import BENCHMARKS


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return bf.build_model(dict(type="CausalLM", vocab_size=320, dim=32, pattern="A", num_layers=2,
                               block_types=dict(attn=dict(mixer=dict(type="GQAttention", heads=2)))))


@BENCHMARKS.register_module(name="toy", force=True)
class Toy(Benchmark):
    """Offline: which letter comes next in 'abcd...'."""

    path, split = "unused", "unused"

    def load(self, max_samples=None, seed=0):
        exs = [Example([("abcdefg"[:k], " " + c) for c in "xyz" + "abcdefg"[k]], 3) for k in range(1, 7)]
        return exs[:max_samples]


# ---------------------------------------------------------------------------------- prompt formats
ROWS = {
    "hellaswag": dict(activity_label="Roof shingle removal", ctx_a="A man is on a roof.", ctx_b="he",
                      endings=["is ripping tiles off. [title] Done", "sits."], label="0"),
    "arc_easy": dict(question="Q?", choices=dict(text=["a", "b"], label=["A", "B"]), answerKey="B"),
    "arc_challenge": dict(question="Q?", choices=dict(text=["a", "b", "c"], label=["1", "2", "3"]), answerKey="3"),
    "piqa": dict(goal="Open a jar", sol1="twist", sol2="kick", label="0"),
    "winogrande": dict(sentence="Sarah beat Maria so _ won.", option1="Sarah", option2="Maria", answer="1"),
    "openbookqa": dict(question_stem="The sun is", choices=dict(text=["hot", "cold"], label=["A", "B"]),
                       answerKey="A"),
    "sciq": dict(support=" Heat rises.", question="What rises?", distractor1="ice", distractor2="rock",
                 distractor3="sand", correct_answer="heat"),
    "boolq": dict(passage="Water is wet.", question="is water wet", answer=True),
    "lambada": dict(text="the cat sat on the mat"),
}


def test_prompt_formats_follow_lm_eval():
    ex = {name: BENCHMARKS.build(dict(type=name)).example(row) for name, row in ROWS.items()}
    assert ex["hellaswag"].requests[0] == ("Roof shingle removal: A man is on a roof. He",
                                           " is ripping tiles off.. Done")
    assert ex["hellaswag"].gold == 0
    assert ex["arc_easy"].requests == [("Question: Q?\nAnswer:", " a"), ("Question: Q?\nAnswer:", " b")]
    assert ex["arc_easy"].gold == 1 and ex["arc_challenge"].gold == 2
    assert ex["piqa"].requests[1] == ("Question: Open a jar\nAnswer:", " kick")
    assert ex["winogrande"].requests == [("Sarah beat Maria so Sarah", " won."), ("Sarah beat Maria so Maria", " won.")]
    assert ex["winogrande"].gold == 0
    assert ex["openbookqa"].requests[0] == ("The sun is", " hot") and ex["openbookqa"].gold == 0
    assert ex["sciq"].requests[3] == ("Heat rises.\nQuestion: What rises?\nAnswer:", " heat") and ex["sciq"].gold == 3
    assert ex["boolq"].requests == [("Water is wet.\nQuestion: is water wet?\nAnswer:", " no"),
                                    ("Water is wet.\nQuestion: is water wet?\nAnswer:", " yes")]
    assert ex["boolq"].gold == 1
    assert ex["lambada"].requests == [("the cat sat on the", " mat")]


def test_mmlu_few_shot_prompt(monkeypatch):
    import brickformer.data.datasets as ds_mod

    def row(q, subject, answer):
        return dict(question=q, subject=subject, choices=["w", "x", "y", "z"], answer=answer)

    dev = [row("D1?", "astronomy", 0), row("D2?", "astronomy", 3), row("D3?", "astronomy", 1),
           row("L1?", "law", 2)]
    test = [row("T?", "astronomy", 2)]

    class Split(list):
        def shuffle(self, seed):
            return self

        def select(self, idx):
            return Split(self[i] for i in idx)

    monkeypatch.setattr(ds_mod, "load_split", lambda path, name, split, *a, **k: Split(dev if split == "dev" else test))
    (ex,) = BENCHMARKS.build(dict(type="mmlu", num_fewshot=2)).load()
    ctx = ex.requests[0][0]
    assert ctx == ("The following are multiple choice questions (with answers) about astronomy.\n\n"
                   "D1?\nA. w\nB. x\nC. y\nD. z\nAnswer: A\n\n"
                   "D2?\nA. w\nB. x\nC. y\nD. z\nAnswer: D\n\n"
                   "T?\nA. w\nB. x\nC. y\nD. z\nAnswer:")  # only the subject's first 2 dev questions
    assert [c for _, c in ex.requests] == [" A", " B", " C", " D"] and ex.gold == 2
    (zero,) = BENCHMARKS.build(dict(type="mmlu", num_fewshot=0)).load()
    assert zero.requests[0][0].endswith("astronomy.\n\nT?\nA. w\nB. x\nC. y\nD. z\nAnswer:")


def test_dataset_settings_can_be_overridden():
    t = BENCHMARKS.build(dict(type="arc_challenge", path="/data/arc", split="validation"))
    assert (t.path, t.name, t.split) == ("/data/arc", "ARC-Challenge", "validation")


# ------------------------------------------------------------------------------------------ scoring
def test_encode_moves_trailing_space_and_keeps_continuation_when_truncating():
    tok = ByteTokenizer()
    ids, n = bm._encode(tok, "ab \n", "cd", max_len=100)
    assert ids == [*b"ab", *b" \ncd"] and n == 4
    assert bm._encode(tok, "ab", " cd", max_len=100, prefix_eos=True) == ([256, *b"ab cd"], 3)
    assert bm._encode(tok, "", "cd", max_len=100) == ([256, *b"cd"], 2)  # empty context: EOS predicts
    ids, n = bm._encode(tok, "x" * 50, " yz", max_len=8)
    assert len(ids) == 9 and ids[-3:] == list(b" yz") and n == 3


class _MergingTokenizer(ByteTokenizer):
    """Merges 'b c' into one token (300), so encoding the pair jointly differs from separately."""

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            if text.startswith("b c", i):
                ids.append(300)
                i += 3
            else:
                ids.append(text.encode()[i] if text[i].isascii() else 0)
                i += 1
        return ids


def test_encode_splits_the_joint_tokenization_like_lm_eval():
    tok = _MergingTokenizer()
    # "xb" + " cd": jointly [x, 300, d], context alone [x, b]: a merge across the boundary, so the
    # continuation is encoded on its own
    assert bm._encode(tok, "xb", " cd", max_len=100) == ([*b"xb", *b" cd"], 3)
    # "x" + "b cd": jointly [x, 300, d] starts with the context's tokens [x], so the continuation is
    # the rest of the joint encoding (the merged token), not the separate encoding [b, ' ', c, d]
    assert bm._encode(tok, "x", "b cd", max_len=100) == ([ord("x"), 300, ord("d")], 2)


def test_batched_scores_match_one_at_a_time():
    model, tok = tiny_model().eval(), ByteTokenizer()
    reqs = [("hello", " world"), ("a much longer context here", " x"), ("", "start"), ("mid ", "dle"),
            ("q", " a fairly long continuation"),
            ("Answer:", " A"), ("Answer:", " B"), ("Answer:", " C")]  # same input: one shared forward pass
    batched = score_requests(model, tok, reqs, batch_tokens=40)
    for req, (lp, greedy) in zip(reqs, batched):
        ids, n = bm._encode(tok, *req, max_len=1024)
        logp = torch.log_softmax(model(torch.tensor([ids[:-1]])).logits[0, -n:].float(), -1)
        assert lp == pytest.approx(logp.gather(-1, torch.tensor(ids[-n:])[:, None]).sum().item(), abs=1e-4)
        assert greedy == bool((logp.argmax(-1) == torch.tensor(ids[-n:])).all())


def test_acc_and_acc_norm(monkeypatch):
    # choice 0: short with the higher total logp; choice 1 (gold): long, lower total, higher per character
    ex = Example([("c", " b"), ("c", " aaaaaaaaaa")], gold=1)
    monkeypatch.setattr(bm, "score_requests", lambda *a, **k: [(-5.0, False), (-6.0, False)])
    task = BENCHMARKS.build(dict(type="arc_easy"))
    assert evaluate_benchmark(None, None, task, [ex]) == dict(acc=0.0, acc_norm=1.0)


def test_lambada_greedy_accuracy_and_perplexity(monkeypatch):
    exs = [Example([("c", " w")], 0), Example([("c", " v")], 0)]
    monkeypatch.setattr(bm, "score_requests", lambda *a, **k: [(-1.0, True), (-3.0, False)])
    res = evaluate_benchmark(None, None, BENCHMARKS.build(dict(type="lambada")), exs)
    assert res["acc"] == 0.5 and res["ppl"] == pytest.approx(math.exp(2.0))


def test_suite_reports_tasks_and_average():
    suite = BenchmarkSuite(tasks=["toy", dict(type="toy", label="toy_small")], max_samples=None)
    suite.tasks["toy_small"].load = lambda max_samples=None, seed=0: Toy().load(3)
    model = tiny_model()
    model.train()
    res = suite.evaluate(model, ByteTokenizer())
    assert model.training  # restored
    assert {"bench/toy_acc", "bench/toy_acc_norm", "bench/toy_small_acc", "bench/avg"} <= set(res)
    assert res["bench/avg"] == pytest.approx((res["bench/toy_acc"] + res["bench/toy_small_acc"]) / 2, abs=1e-5)
    with pytest.raises(ValueError, match="twice"):
        BenchmarkSuite(tasks=["toy", "toy"])
    with pytest.raises(KeyError, match="not registered"):
        BenchmarkSuite(tasks=["no_such_task"])


# -------------------------------------------------------------------------------------- integration
def test_benchmarks_run_during_and_after_training(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from brickformer.train import build_experiment

    text = tmp_path / "train.txt"
    text.write_text("".join(f"abcdefg {i}\n\n" for i in range(200)))
    cfg = dict(
        model=dict(type="CausalLM", vocab_size="auto", dim=32, pattern="A", num_layers=2,
                   block_types=dict(attn=dict(mixer=dict(type="GQAttention", heads=2)))),
        tokenizer=dict(type="ByteTokenizer"),
        data=dict(train=dict(path="text", data_files=str(text), seq_len=32)),
        train=dict(type="HFTrainer", output_dir=str(tmp_path / "run"), max_steps=5, per_device_train_batch_size=4,
                   learning_rate=1e-3, logging_steps=1, save_steps=100, use_cpu=True, disable_tqdm=True, seed=0),
        benchmarks=dict(tasks=["toy"], every=2),
    )
    exp = build_experiment(cfg)
    trainer = exp.run()
    lines = [json.loads(x) for x in (tmp_path / "run" / "benchmarks.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [2, 4, 5]  # every 2 steps, then the end
    logged = [h for h in trainer.state.log_history if "bench/toy_acc" in h]
    assert [h["step"] for h in logged] == [2, 4, 5] and all("bench/avg" in h for h in logged)
