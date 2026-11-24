"""Config-driven training through the HF Trainer, fully offline (byte tokenizer + local text files)."""

import json

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")

import brickformer as bf  # noqa: E402
from brickformer.data import ByteTokenizer, CausalLMCollator, HFTextDataset  # noqa: E402
from brickformer.train import build_experiment, run_experiment  # noqa: E402
from brickformer.train.cli import main as cli_main  # noqa: E402

TEXT = "".join(f"Story {i}: the little model learned to count to {i}.\n\n" for i in range(300))


@pytest.fixture
def text_files(tmp_path):
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text(TEXT)
    val.write_text(TEXT[:3000])
    return str(train), str(val)


def experiment_cfg(tmp_path, text_files, **train_kw):
    train_file, val_file = text_files
    data = dict(type="HFTextDataset", path="text", load_kwargs=dict(sample_by="paragraph"), seq_len=64)
    train = dict(type="HFTrainer", output_dir=str(tmp_path / "run"), max_steps=6, per_device_train_batch_size=4,
                 learning_rate=3e-3, warmup_steps=1, logging_steps=2, save_steps=3, use_cpu=True,
                 disable_tqdm=True, seed=0)
    train.update(train_kw)
    return dict(
        model=dict(
            type="CausalLM", vocab_size="auto", dim=64, pattern="G*", tie_embeddings=True,
            block_types=dict(
                attn=dict(mixer=dict(type="GQAttention", heads=4, kv_heads=2), ffn=dict(type="SwiGLU", hidden=128)),
                gdn=dict(mixer=dict(type="GatedDeltaNet", heads=2, backend="torch"),
                         ffn=dict(type="MoE", num_experts=4, top_k=2, hidden=32)),
            ),
        ),
        tokenizer=dict(type="ByteTokenizer"),
        data=dict(train=dict(data, data_files=train_file), val=dict(data, data_files=val_file)),
        train=train,
    )


# ---------------------------------------------------------------- data
def test_packing_and_collator(text_files):
    tok = ByteTokenizer()
    ds = HFTextDataset("text", tok, seq_len=32, data_files=text_files[0], load_kwargs=dict(sample_by="paragraph"))
    assert all(len(x) == 32 for x in ds["input_ids"])
    flat = [t for x in ds["input_ids"] for t in x]
    assert flat.count(tok.eos_token_id) > 0  # documents separated by EOS
    assert tok.decode(flat[:20]) == TEXT[:20]

    batch = CausalLMCollator(pad_token_id=0)([{"input_ids": [5, 6, 7]}, {"input_ids": [8]}])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"].tolist() == [[5, 6, 7], [8, -100, -100]]


def test_streaming_matches_map_style(text_files):
    tok = ByteTokenizer()
    kw = dict(data_files=text_files[0], load_kwargs=dict(sample_by="paragraph"))
    eager = HFTextDataset("text", tok, seq_len=48, **kw)
    stream = HFTextDataset("text", tok, seq_len=48, streaming=True, **kw)
    blocks = [x["input_ids"].tolist() for x in stream]
    # map-style packs per 1000-doc batch, so the streams agree on every complete leading block
    assert blocks[:50] == eager["input_ids"][:50]
    assert len(blocks) >= len(eager)


def test_vocab_auto_and_validation(tmp_path, text_files):
    exp = build_experiment(experiment_cfg(tmp_path, text_files), build_data=False)
    assert exp.config["model"]["vocab_size"] == 320  # 257 rounded up to a multiple of 64
    cfg = experiment_cfg(tmp_path, text_files)
    cfg["model"]["vocab_size"] = 100
    with pytest.raises(ValueError, match="smaller than"):
        build_experiment(cfg, build_data=False)
    with pytest.raises(KeyError, match="missing sections"):
        build_experiment({"model": cfg["model"]})


def test_warmup_ratio_accepted_on_any_transformers_version(tmp_path):
    from brickformer.train.hf_trainer import HFTrainer

    args = HFTrainer(output_dir=str(tmp_path), warmup_ratio=0.1, max_steps=100, use_cpu=True).make_args(False)
    assert getattr(args, "warmup_ratio", None) == 0.1 or args.warmup_steps == 0.1


def test_overrides():
    cfg = dict(train=dict(lr=1), model=dict(pattern=["a", "b"], block_types=dict(a=dict(mixer=dict(heads=4)))))
    out = bf.apply_overrides(cfg, ["train.lr=1e-3", "model.pattern.1=c", "model.block_types.a.mixer.heads=8",
                                   "train.bf16=true", "train.name=run-1", "train.new.deep=[1, 2]"])
    assert out["train"] == dict(lr=1e-3, bf16=True, name="run-1", new=dict(deep=[1, 2]))
    assert out["model"]["pattern"] == ["a", "c"]
    assert out["model"]["block_types"]["a"]["mixer"]["heads"] == 8
    assert cfg["train"]["lr"] == 1  # input untouched


def test_output_survives_mapping_rebuild():
    """accelerate's mixed-precision wrapper rebuilds dict outputs as type(out)({...})."""
    from brickformer.models import CausalLMOutput

    out = CausalLMOutput(logits=torch.zeros(1), loss=torch.ones(()))
    rebuilt = type(out)({k: v for k, v in out.items()})
    assert rebuilt.loss is out.loss and rebuilt.logits is out.logits and rebuilt.aux_loss is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 backward needs CUDA (not all CPUs support it)")
def test_bf16_mixed_precision_with_grad_accumulation(tmp_path, text_files):
    cfg = experiment_cfg(tmp_path, text_files, max_steps=2, bf16=True, gradient_accumulation_steps=2,
                         save_steps=100, use_cpu=False)
    cfg["data"]["val"] = None
    trainer = run_experiment(cfg)
    assert trainer.state.global_step == 2
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in trainer.state.log_history if "loss" in h)


# ---------------------------------------------------------------- end-to-end training
def test_train_checkpoint_resume(tmp_path, text_files, monkeypatch):
    cfg = experiment_cfg(tmp_path, text_files)
    trainer = run_experiment(cfg)
    run = tmp_path / "run"
    history = trainer.state.log_history
    losses = [h["loss"] for h in history if "loss" in h]
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))
    assert any("aux_loss" in h for h in history)  # MoE balance loss logged separately
    assert any("eval_loss" in h for h in history)

    # checkpoints use the BrickFormer format, tied embeddings included, and reload exactly
    for ckpt in (run / "checkpoint-3", run / "checkpoint-6", run):
        assert (ckpt / "model.pt").is_file() and (ckpt / "experiment.json").is_file()
    exp_cfg = json.loads((run / "experiment.json").read_text())
    assert exp_cfg["model"]["vocab_size"] == 320 and exp_cfg["tokenizer"]["type"] == "ByteTokenizer"
    loaded = bf.load_pretrained(run).eval()
    assert loaded.lm_head.weight is loaded.embed.weight
    model = trainer.model.eval().cpu()
    ids = torch.randint(0, 257, (1, 16))
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits)

    # resume from the step-3 checkpoint and continue to step 8
    from brickformer.train.hf_trainer import BrickTrainer

    steps = []
    original = BrickTrainer.training_step

    def counting_step(self, *args, **kwargs):
        steps.append(self.state.global_step)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BrickTrainer, "training_step", counting_step)
    cfg2 = experiment_cfg(tmp_path, text_files, max_steps=8, resume_from_checkpoint=str(run / "checkpoint-3"))
    trainer2 = run_experiment(cfg2)
    assert trainer2.state.global_step == 8
    assert steps == [3, 4, 5, 6, 7]  # continued from step 3 instead of restarting


def test_cli_dry_run_and_train(tmp_path, text_files, capsys):
    cfg_path = tmp_path / "exp.py"
    bf.dump_config(experiment_cfg(tmp_path, text_files), cfg_path)
    cli_main([str(cfg_path), "--dry-run", "--cfg-options", "model.dim=32", "train.max_steps=2"])
    out = capsys.readouterr().out
    assert '"dim": 32' in out and "total" in out
    cli_main([str(cfg_path), "--cfg-options", "train.max_steps=2", "train.save_steps=100", "data.val=None"])
    assert (tmp_path / "run" / "model.pt").is_file()


@pytest.mark.parametrize("path", sorted((CONFIGS_DIR := __import__("pathlib").Path(__file__).parent.parent
                                          / "configs" / "train").glob("*.py")), ids=lambda p: p.name)
def test_training_configs_are_valid(path, tmp_path):
    """Offline check of every experiment config: objective, dataset types and TrainingArguments."""
    from brickformer.train.experiment import _dataset_cfg
    from brickformer.train.hf_trainer import HFTrainer
    from brickformer.train.objectives import build_objective

    cfg = bf.load_config(path)
    objective = build_objective(cfg.get("objective"))
    for split in ("train", "val"):
        if cfg["data"].get(split) is not None:
            _dataset_cfg(cfg["data"][split], objective)
    train = dict(cfg["train"], output_dir=str(tmp_path))
    train.pop("type")
    HFTrainer(**train).make_args(has_eval="val" in cfg["data"])
    assert "model" in cfg or "init_from" in cfg
