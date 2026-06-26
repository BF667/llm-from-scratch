"""
llm_scratch — Train a small decoder-only LLM from scratch on Google Colab.
从头训练小型 decoder-only LLM，支持 Google Colab。

Public API:
    build_model(arch, size) -> nn.Module  (returns a HF-style model)
    build_config(arch, size, **overrides) -> PretrainedConfig
    list_presets() -> dict
    run_training(...)
    load_wikitext_dataset / load_custom_dataset / build_tokenizer
    GPTScratchForCausalLM / LlamaScratchForCausalLM

Note: submodules that depend on `torch` / `transformers` (model_gpt2,
model_llama, dataset, train, generate) are imported **lazily** via
`__getattr__`. This means `from llm_scratch.config import build_config`
works even if `torch` itself fails to import (which happens on some
Colab runtimes when transformers / torch / torchvision versions are out
of sync — see Colab notebook section "0. Fix torch (if needed)").

注意：依赖 `torch` / `transformers` 的子模块（model_gpt2、model_llama、
dataset、train、generate）通过 `__getattr__` 懒加载。这样即使 `torch` 自身
无法 import（Colab 上偶发的 transformers/torch 版本不一致问题），仍可使用
`from llm_scratch.config import build_config`。
"""

# config.py is pure stdlib — always safe to import eagerly.
from .config import build_config, list_presets, SIZE_PRESETS, estimate_params

__all__ = [
    "build_config",
    "build_model",
    "estimate_params",
    "list_presets",
    "SIZE_PRESETS",
    "GPTScratchForCausalLM",
    "LlamaScratchForCausalLM",
    "load_wikitext_dataset",
    "load_custom_dataset",
    "build_tokenizer",
    "run_training",
]


def __getattr__(name: str):
    """Lazy-import torch-dependent submodules on first access.
    在首次访问时懒加载依赖 torch 的子模块。"""
    if name == "build_model":
        from .model_gpt2 import GPTScratchForCausalLM
        from .model_llama import LlamaScratchForCausalLM

        def build_model(arch: str = "gpt2", size: str = "tiny", **overrides):
            """Build a from-scratch LLM by architecture + size preset.
            按架构 + 尺寸预设构建一个从头训练的 LLM。"""
            cfg = build_config(arch=arch, size=size, **overrides)
            arch_lc = arch.lower()
            if arch_lc == "gpt2":
                return GPTScratchForCausalLM(cfg)
            elif arch_lc == "llama":
                return LlamaScratchForCausalLM(cfg)
            else:
                raise ValueError(f"Unknown arch: {arch!r}. Use 'gpt2' or 'llama'.")

        globals()["build_model"] = build_model
        return build_model

    if name == "GPTScratchForCausalLM":
        from .model_gpt2 import GPTScratchForCausalLM as _M
        globals()["GPTScratchForCausalLM"] = _M
        return _M

    if name == "LlamaScratchForCausalLM":
        from .model_llama import LlamaScratchForCausalLM as _M
        globals()["LlamaScratchForCausalLM"] = _M
        return _M

    if name == "load_wikitext_dataset":
        from .dataset import load_wikitext_dataset as _F
        globals()["load_wikitext_dataset"] = _F
        return _F

    if name == "load_custom_dataset":
        from .dataset import load_custom_dataset as _F
        globals()["load_custom_dataset"] = _F
        return _F

    if name == "build_tokenizer":
        from .dataset import build_tokenizer as _F
        globals()["build_tokenizer"] = _F
        return _F

    if name == "run_training":
        from .train import run_training as _F
        globals()["run_training"] = _F
        return _F

    raise AttributeError(f"module 'llm_scratch' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
