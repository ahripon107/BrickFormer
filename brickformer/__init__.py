"""BrickFormer: build LLMs from mixed block types, defined entirely in config."""

from . import ffns, mixers, norms, positional  # noqa: F401  (registers components)
from .block import Block
from .config import build_from_file, dump_config, load_config, load_pretrained, save_pretrained
from .generation import generate, sample_next
from .models import CausalLM, CausalLMOutput, ModelCache
from .registry import BLOCKS, FFNS, MIXERS, MODELS, NORMS, POS_EMBEDS, Registry, build_model
from .utils import count_params, estimate_flops, summary

__version__ = "0.1.0"

__all__ = [
    "Registry", "MIXERS", "FFNS", "NORMS", "POS_EMBEDS", "BLOCKS", "MODELS", "build_model",
    "Block", "CausalLM", "CausalLMOutput", "ModelCache", "generate", "sample_next",
    "load_config", "dump_config", "build_from_file", "save_pretrained", "load_pretrained",
    "count_params", "estimate_flops", "summary",
]
