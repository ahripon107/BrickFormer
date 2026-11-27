"""Data-mixing schedules for pretraining (offline: byte tokenizer + local text files)."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("datasets")

from brickformer.data import ByteTokenizer, CausalLMCollator, MixturePretrainDataset, MixtureSchedule  # noqa: E402

SEQ = 16


@pytest.fixture
def sources(tmp_path):
    """Three sources whose blocks are recognisable by their letter."""
    out = {}
    for name, letter, n in (("web", "w", 400), ("math", "m", 400), ("code", "c", 40)):
        f = tmp_path / f"{name}.txt"
        f.write_text("".join(f"{letter * 30} {i}\n\n" for i in range(n)))
        out[name] = dict(path="text", data_files=str(f), load_kwargs=dict(sample_by="paragraph"))
    return out


def _letters(ds, n):
    out = []
    for _, ex in zip(range(n), ds):
        text = bytes(t for t in ex["input_ids"].tolist() if t < 256).decode()
        letter = next(c for c in text if c in "wmc")
        assert ds.source_names[ex["source"]][0] == letter  # the source index matches the content
        out.append(letter)
    return out


# ---------------------------------------------------------------- schedule maths
def test_schedule_step_phases_and_tokens():
    s = MixtureSchedule(["a", "b"], schedule=[dict(until=0.5, weights=dict(a=1)),
                                              dict(until=1.0, weights=dict(a=1, b=3))], total_blocks=100)
    assert s.as_dict(0) == dict(a=1.0, b=0.0)
    assert s.as_dict(49) == dict(a=1.0, b=0.0)
    assert s.as_dict(50) == dict(a=0.25, b=0.75)
    assert s.as_dict(10_000) == dict(a=0.25, b=0.75)  # past the end: the last phase continues
    t = MixtureSchedule(["a", "b"], schedule=[dict(until_tokens=800, weights=dict(a=1)),
                                              dict(until_tokens=1600, weights=dict(b=1))], seq_len=16)
    assert t.as_dict(49) == dict(a=1.0, b=0.0) and t.as_dict(50) == dict(a=0.0, b=1.0)


def test_schedule_linear_keypoints():
    s = MixtureSchedule(["a", "b"], schedule=[dict(at=0.2, weights=dict(a=1)), dict(at=0.6, weights=dict(b=1))],
                        total_blocks=100)
    assert s.as_dict(0) == dict(a=1.0, b=0.0)
    np.testing.assert_allclose(s.weights_at(40), [0.5, 0.5])
    assert s.as_dict(90) == dict(a=0.0, b=1.0)


@pytest.mark.parametrize("kw, err", [
    (dict(weights=dict(a=1, z=1)), "unknown sources"),
    (dict(weights=dict(a=0)), "positive sum"),
    (dict(schedule=[dict(until=0.5, weights=dict(a=1)), dict(at=1.0, weights=dict(b=1))]), "all be step"),
    (dict(schedule=[dict(until=0.5, weights=dict(a=1))], total_blocks=None), "needs the run length"),
    (dict(schedule=[dict(until=0.8, weights=dict(a=1)), dict(until=0.4, weights=dict(b=1))]), "must increase"),
    (dict(schedule=[dict(until=50, weights=dict(a=1))]), "fraction"),
    (dict(), "exactly one"),
])
def test_schedule_validation(kw, err):
    kw.setdefault("total_blocks", 100)
    with pytest.raises((ValueError, KeyError), match=err):
        MixtureSchedule(["a", "b"], **kw)


# ---------------------------------------------------------------- dataset
def test_static_mixture_proportions(sources):
    ds = MixturePretrainDataset(dict(web=sources["web"], math=sources["math"]), ByteTokenizer(), SEQ,
                                weights=dict(web=3, math=1), seed=0)
    letters = _letters(ds, 800)
    assert 0.70 < letters.count("w") / len(letters) < 0.80


def test_step_schedule_switches_sources(sources):
    ds = MixturePretrainDataset(sources, ByteTokenizer(), SEQ, total_blocks=200, schedule=[
        dict(until=0.25, weights=dict(web=1)),
        dict(until=1.0, weights=dict(math=1)),
    ])
    letters = _letters(ds, 200)
    assert set(letters[:50]) == {"w"} and set(letters[50:]) == {"m"}


def test_linear_schedule_ramps(sources):
    ds = MixturePretrainDataset(dict(web=sources["web"], math=sources["math"]), ByteTokenizer(), SEQ,
                                total_blocks=1000, seed=1,
                                schedule=[dict(at=0.0, weights=dict(web=1)), dict(at=1.0, weights=dict(math=1))])
    letters = _letters(ds, 1000)
    share = [letters[i:i + 250].count("m") / 250 for i in range(0, 1000, 250)]
    assert share == sorted(share) and share[0] < 0.25 and share[-1] > 0.75


def test_reproducible_and_worker_independent(sources):
    def make():
        return MixturePretrainDataset(sources, ByteTokenizer(), SEQ, weights=dict(web=2, math=2, code=1), seed=3)

    first = [ex["input_ids"].tolist() for _, ex in zip(range(120), make())]
    again = [ex["input_ids"].tolist() for _, ex in zip(range(120), make())]
    assert first == again
    loader = torch.utils.data.DataLoader(make(), batch_size=None, num_workers=2)
    sources_seen = [int(ex["source"]) for _, ex in zip(range(120), loader)]
    assert sources_seen == [int(ex["source"]) for _, ex in zip(range(120), make())]


def test_exhausted_source_repeats_or_stops(sources):
    # "code" has ~40 short documents: a few dozen blocks, far fewer than asked for
    ds = MixturePretrainDataset(dict(code=sources["code"]), ByteTokenizer(), SEQ, weights=dict(code=1))
    assert len(_letters(ds, 300)) == 300
    stop = MixturePretrainDataset(dict(code=sources["code"]), ByteTokenizer(), SEQ, weights=dict(code=1),
                                  on_exhausted="stop")
    assert 0 < len(list(stop)) < 300


def test_collator_passes_source_ids(sources):
    ds = MixturePretrainDataset(sources, ByteTokenizer(), SEQ, weights=dict(web=1, math=1))
    batch = CausalLMCollator(0)([ex for _, ex in zip(range(4), ds)])
    assert batch["input_ids"].shape == (4, SEQ) and batch["source_ids"].shape == (4,)


# ---------------------------------------------------------------- experiment / trainer
def test_experiment_fills_run_length_and_logs_mixture(tmp_path, sources):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from brickformer.train import build_experiment, run_experiment

    val = {name: dict(path="text", data_files=src["data_files"], load_kwargs=src["load_kwargs"], seq_len=SEQ,
                      max_samples=20) for name, src in sources.items() if name != "code"}
    cfg = dict(
        model=dict(type="CausalLM", vocab_size="auto", dim=32, pattern="A", num_layers=2,
                   block_types=dict(attn=dict(mixer=dict(type="GQAttention", heads=2)))),
        tokenizer=dict(type="ByteTokenizer"),
        data=dict(
            train=dict(type="MixturePretrainDataset", seq_len=SEQ, sources=sources, schedule=[
                dict(until=0.5, weights=dict(web=1)),
                dict(until=1.0, weights=dict(math=1, code=1)),
            ]),
            val=val,
        ),
        train=dict(type="HFTrainer", output_dir=str(tmp_path / "run"), max_steps=6, per_device_train_batch_size=4,
                   gradient_accumulation_steps=2, learning_rate=1e-3, logging_steps=3, eval_steps=6, save_steps=6,
                   use_cpu=True, disable_tqdm=True, seed=0),
    )
    exp = build_experiment(cfg)
    assert exp.train_dataset.schedule.points == [0.5 * 6 * 4 * 2, 1.0 * 6 * 4 * 2]  # max_steps x batch x accum
    assert "total_blocks" not in exp.config["data"]["train"]  # recomputed per run, so not saved

    run_experiment(cfg)
    history = json.loads((tmp_path / "run" / "checkpoint-6" / "trainer_state.json").read_text())["log_history"]
    first, second = [h for h in history if "loss" in h and "eval_loss" not in h][:2]
    assert first["mix/web"] == 1.0 and first["mix/math"] == 0.0  # steps 1-3: web only
    assert second["mix/web"] == 0.0 and second["mix/math"] + second["mix/code"] == pytest.approx(1.0)
    evals = [h for h in history if any(k.startswith("eval_") for k in h)]
    assert any("eval_web_loss" in h for h in evals) and any("eval_math_loss" in h for h in evals)
    assert not any(k.startswith("eval_mix") for h in evals for k in h)
