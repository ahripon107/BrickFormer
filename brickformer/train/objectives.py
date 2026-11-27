"""Training objectives. The experiment's ``objective`` section picks one::

    objective = 'pretrain'                                   # or 'sft'
    objective = dict(type='dpo', beta=0.1)
    objective = dict(type='grpo', num_generations=8, max_completion_len=256,
                     rewards=[dict(type='NumberMatch', reference_field='answer')])

An objective decides which dataset type the data section defaults to (and which it accepts), how
batches are collated, and the loss. The runner calls ``objective.compute_loss(trainer, model, inputs)``.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ..config import load_pretrained
from ..data.datasets import CausalLMCollator, PreferenceCollator, PromptCollator, _pad
from ..generation import generate
from ..registry import OBJECTIVES, REWARDS
from . import rewards as _rewards  # noqa: F401  (registers built-in rewards)


def token_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of ``labels`` (aligned with the inputs, shifted here) and their mask.
    Returns ``(logp [N, L-1] zeroed where masked, mask [N, L-1])``."""
    logits = logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != -100
    picked = logits.gather(-1, targets.clamp(min=0)[..., None]).squeeze(-1)
    logp = picked - torch.logsumexp(logits, dim=-1)
    return logp * mask, mask


class Objective:
    name: str = ""
    default_dataset: str = ""
    needs_ref_model: bool = False
    ref_model: str | None = None

    def collator(self, tokenizer):
        return CausalLMCollator(tokenizer.pad_token_id)

    def build_ref_model(self, model):
        """Frozen reference: a checkpoint (``ref_model='path'``) or a copy of the starting policy."""
        ref = load_pretrained(self.ref_model) if self.ref_model else copy.deepcopy(model)
        ref.eval().requires_grad_(False)
        return ref

    source_names: list[str] | None = None  # set by the experiment for a MixturePretrainDataset

    def record_mixture(self, trainer, source_ids: torch.Tensor) -> None:
        """Logs each source's share of the training rows as ``mix/<name>``."""
        if not trainer.model.training:
            return
        names = self.source_names or [str(i) for i in range(int(source_ids.max()) + 1)]
        for i, name in enumerate(names):
            trainer.record(f"mix/{name}", (source_ids == i).float().mean())

    def compute_loss(self, trainer, model, inputs) -> torch.Tensor:
        source_ids = inputs.pop("source_ids", None)
        if source_ids is not None:
            self.record_mixture(trainer, source_ids)
        out = model(**inputs)
        if out.aux_loss is not None:
            trainer.record("aux_loss", out.aux_loss)
        return out.loss


@OBJECTIVES.register_module(name="pretrain")
class Pretrain(Objective):
    """Next-token prediction on packed text."""

    name, default_dataset = "pretrain", "PretrainDataset"


@OBJECTIVES.register_module(name="sft")
class SFT(Objective):
    """Next-token prediction on responses only (prompt tokens are masked by ``SFTDataset``)."""

    name, default_dataset = "sft", "SFTDataset"


@OBJECTIVES.register_module(name="dpo")
class DPO(Objective):
    """Direct Preference Optimization (Rafailov et al., 2023) against a frozen reference model.

    ``loss_type``: ``'sigmoid'`` (DPO; ``label_smoothing`` > 0 gives conservative DPO) or ``'ipo'``.
    ``ref_model``: checkpoint directory; defaults to a frozen copy of the starting policy.
    """

    name, default_dataset, needs_ref_model = "dpo", "PreferenceDataset", True

    def __init__(self, beta: float = 0.1, loss_type: str = "sigmoid", label_smoothing: float = 0.0,
                 ref_model: str | None = None):
        if loss_type not in ("sigmoid", "ipo"):
            raise ValueError(f"DPO loss_type must be 'sigmoid' or 'ipo', got {loss_type!r}")
        self.beta, self.loss_type, self.label_smoothing, self.ref_model = beta, loss_type, label_smoothing, ref_model

    def collator(self, tokenizer):
        return PreferenceCollator(tokenizer.pad_token_id)

    def compute_loss(self, trainer, model, inputs):
        ids, labels = inputs["input_ids"], inputs["labels"]
        out = model(input_ids=ids)
        policy = token_logps(out.logits, labels)[0].sum(-1)
        with torch.no_grad():
            ref = token_logps(trainer.ref_model(input_ids=ids).logits, labels)[0].sum(-1)
        pc, pr = policy.chunk(2)
        rc, rr = ref.chunk(2)
        margin = (pc - rc) - (pr - rr)
        if self.loss_type == "sigmoid":
            ls = self.label_smoothing
            losses = -(1 - ls) * F.logsigmoid(self.beta * margin) - ls * F.logsigmoid(-self.beta * margin)
        else:
            losses = (margin - 1 / (2 * self.beta)) ** 2
        loss = losses.mean()
        chosen_reward, rejected_reward = self.beta * (pc - rc).detach(), self.beta * (pr - rr).detach()
        trainer.record("rewards/chosen", chosen_reward.mean())
        trainer.record("rewards/rejected", rejected_reward.mean())
        trainer.record("rewards/accuracy", (chosen_reward > rejected_reward).float().mean())
        trainer.record("rewards/margin", (chosen_reward - rejected_reward).mean())
        if out.aux_loss is not None:
            trainer.record("aux_loss", out.aux_loss)
            if model.training:
                loss = loss + out.aux_loss
        return loss


@OBJECTIVES.register_module(name="grpo")
class GRPO(Objective):
    """Group Relative Policy Optimization (Shao et al., 2024), on-policy (one update per batch of
    generations, so the importance ratio is 1 and no clipping is needed).

    For each prompt, ``num_generations`` completions are sampled with BrickFormer's ``generate()``,
    scored with the weighted ``rewards``, and given advantage ``(r - mean_group) / std_group``
    (``scale_rewards=False`` drops the std, as in Dr. GRPO). ``beta > 0`` adds the k3 KL penalty
    to a frozen reference model. ``loss_agg``: ``'token'`` averages over all completion tokens in
    the batch (DAPO); ``'sequence'`` averages per completion first (original GRPO).

    ``per_device_train_batch_size`` counts prompts, so each step trains on
    ``batch_size * num_generations`` completions.
    """

    name, default_dataset = "grpo", "PromptDataset"

    def __init__(self, rewards: list[dict], num_generations: int = 8, max_completion_len: int = 256,
                 temperature: float = 1.0, top_p: float = 1.0, top_k: int | None = None, beta: float = 0.0,
                 scale_rewards: bool = True, loss_agg: str = "token", ref_model: str | None = None):
        if num_generations < 2:
            raise ValueError("GRPO needs num_generations >= 2 to compare completions within a group")
        if temperature <= 0:
            raise ValueError("GRPO samples completions, so temperature must be > 0")
        if loss_agg not in ("token", "sequence"):
            raise ValueError(f"loss_agg must be 'token' or 'sequence', got {loss_agg!r}")
        if not rewards:
            raise ValueError("GRPO needs at least one reward, e.g. rewards=[dict(type='NumberMatch')]")
        self.rewards = [REWARDS.build(r) for r in rewards]
        self.num_generations, self.max_completion_len = num_generations, max_completion_len
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.beta, self.scale_rewards, self.loss_agg, self.ref_model = beta, scale_rewards, loss_agg, ref_model
        self.needs_ref_model = beta > 0

    def collator(self, tokenizer):
        return PromptCollator()

    @torch.no_grad()
    def rollout(self, trainer, model, prompts: list[list[int]]):
        """Sample ``num_generations`` completions per prompt. Completions end at (and include) the
        first EOS, or at ``max_completion_len``."""
        tok, g = trainer.bf_tokenizer, self.num_generations
        eos, device = tok.eos_token_id, trainer.args.device
        core = trainer.accelerator.unwrap_model(model)
        seqs, labels, texts, lengths, truncated = [], [], [], [], []
        for p in prompts:
            out = generate(core, torch.tensor([p] * g, device=device), self.max_completion_len,
                           temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, eos_token_id=eos)
            for c in out[:, len(p):].tolist():
                ended = eos in c
                c = c[:c.index(eos) + 1] if ended else c
                seqs.append(p + c)
                labels.append([-100] * len(p) + c)
                texts.append(tok.decode(c[:-1] if ended else c))
                lengths.append(len(c))
                truncated.append(not ended)
        return seqs, labels, texts, lengths, truncated

    def score(self, trainer, prompts: list[list[int]], texts: list[str], columns: dict) -> torch.Tensor:
        g = self.num_generations
        prompt_texts = [trainer.bf_tokenizer.decode(p) for p in prompts for _ in range(g)]
        cols = {k: [v for v in vals for _ in range(g)] for k, vals in columns.items()}
        total = torch.zeros(len(texts))
        for reward in self.rewards:
            values = torch.tensor(reward(prompt_texts, texts, **cols), dtype=torch.float32)
            if values.shape != total.shape:
                raise ValueError(f"reward {reward.name} returned {values.numel()} values for {len(texts)} completions")
            trainer.record(f"rewards/{reward.name}", values.mean())
            total += reward.weight * values
        return total

    def compute_loss(self, trainer, model, inputs):
        prompts, columns = inputs["prompt_ids"], inputs.get("columns", {})
        seqs, labels, texts, lengths, truncated = self.rollout(trainer, model, prompts)
        rewards = self.score(trainer, prompts, texts, columns).view(-1, self.num_generations)
        std = rewards.std(dim=1, keepdim=True)
        adv = rewards - rewards.mean(dim=1, keepdim=True)
        if self.scale_rewards:
            adv = adv / (std + 1e-4)
        adv = adv.view(-1).to(trainer.args.device)

        batch = {k: v.to(trainer.args.device) for k, v in _pad(seqs, labels, trainer.bf_tokenizer.pad_token_id).items()}
        out = model(input_ids=batch["input_ids"])
        logp, mask = token_logps(out.logits, batch["labels"])
        per_token = -torch.exp(logp - logp.detach()) * adv[:, None]  # value 1 * -A; gradient -A * grad(logp)
        if self.beta > 0:
            with torch.no_grad():
                ref_logp = token_logps(trainer.ref_model(input_ids=batch["input_ids"]).logits, batch["labels"])[0]
            delta = ref_logp - logp
            kl = torch.exp(delta) - delta - 1
            per_token = per_token + self.beta * kl
            trainer.record("kl", (kl * mask).sum() / mask.sum())
        maskf = mask.float()
        if self.loss_agg == "token":
            loss = (per_token * maskf).sum() / maskf.sum().clamp(min=1)
        else:
            loss = ((per_token * maskf).sum(1) / maskf.sum(1).clamp(min=1)).mean()
        if out.aux_loss is not None:
            trainer.record("aux_loss", out.aux_loss)
            if model.training:
                loss = loss + out.aux_loss

        trainer.record("reward", rewards.mean())
        trainer.record("reward_std", std.mean())
        trainer.record("frac_reward_zero_std", (std == 0).float().mean())
        trainer.record("completion_length", torch.tensor(lengths, dtype=torch.float32).mean())
        trainer.record("frac_truncated", torch.tensor(truncated, dtype=torch.float32).mean())
        return loss


def build_objective(cfg) -> Objective:
    if cfg is None:
        cfg = "pretrain"
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    return OBJECTIVES.build(cfg)
