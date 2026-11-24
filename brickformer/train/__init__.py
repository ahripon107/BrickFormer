"""Config-driven training (optional; needs ``pip install brickformer[train]``)."""

from .experiment import Experiment, build_experiment, resolve_vocab_size, run_experiment
from .hf_trainer import HFTrainer
from .objectives import DPO, GRPO, SFT, Objective, Pretrain, build_objective, token_logps
from .rewards import ExactMatch, LengthPenalty, NumberMatch, PythonReward, RegexMatch, Reward

__all__ = [
    "Experiment", "build_experiment", "run_experiment", "resolve_vocab_size", "HFTrainer",
    "Objective", "Pretrain", "SFT", "DPO", "GRPO", "build_objective", "token_logps",
    "Reward", "NumberMatch", "ExactMatch", "RegexMatch", "LengthPenalty", "PythonReward",
]
