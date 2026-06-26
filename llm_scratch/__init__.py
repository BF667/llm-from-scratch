"""
llm_scratch — Train a small decoder-only LLM from scratch on Google Colab.
从头训练小型 decoder-only LLM，支持 Google Colab。

Public API:
    build_model(arch, size) -> nn.Module  (returns a HF-style model)
    build_config(arch, size, **overrides) -> PretrainedConfig
    list_presets() -> dict
"""

from .config import build_config, list_presets, SIZE_PRESETS
from .model_gpt2 import GPTScratchForCausalLM
from .model_llama import LlamaScratchForCausalLM
from .dataset import load_wikitext_dataset, build_tokenizer
from .train import run_training

__all__ = [
    "build_config",
    "build_model",
    "list_presets",
    "SIZE_PRESETS",
    "GPTScratchForCausalLM",
    "LlamaScratchForCausalLM",
    "load_wikitext_dataset",
    "build_tokenizer",
    "run_training",
]


def build_model(arch: str = "gpt2", size: str = "tiny", **overrides):
    """Build a from-scratch LLM by architecture + size preset.
    按架构 + 尺寸预设构建一个从头训练的 LLM。
    """
    cfg = build_config(arch=arch, size=size, **overrides)
    arch = arch.lower()
    if arch == "gpt2":
        return GPTScratchForCausalLM(cfg)
    elif arch == "llama":
        return LlamaScratchForCausalLM(cfg)
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Use 'gpt2' or 'llama'.")
