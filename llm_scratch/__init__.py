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

# ---------------------------------------------------------------------------
# Torch._dynamo Config() compat shim — runs at import time.
# ---------------------------------------------------------------------------
# Some Colab runtimes ship a `torch` whose `_dynamo/config.py` calls
#     Config(default=True, deprecated=True, deprecation_message="...")
# but whose `torch.utils._config_typing.Config` dataclass doesn't accept
# the `deprecated` / `deprecation_message` kwargs. The result is:
#     TypeError: Config() got an unexpected keyword argument 'deprecated'
# This fires the moment transformers/datasets/accelerate pulls in
# torch._dynamo.config — which `llm_scratch.dataset` / `llm_scratch.train`
# do indirectly.
#
# We patch `Config.__init__` to silently drop unknown kwargs BEFORE any
# submodule import triggers torch._dynamo.config. This is a no-op on
# healthy torch installs (the patch wraps __init__ but only filters
# kwargs that aren't in the signature anyway).
#
# 部分Colab运行时的 torch._dynamo/config.py 调用了
#     Config(default=True, deprecated=True, deprecation_message="...")
# 但 torch.utils._config_typing.Config 不接受 deprecated / deprecation_message
# 这两个 kwarg。一旦 transformers/datasets/accelerate 间接 import
# torch._dynamo.config 就会抛 TypeError。这里在 import 子模块之前先 patch
# Config.__init__，让它静默丢弃未知 kwarg。健康环境下该 patch 是 no-op。
def _apply_torch_dynamo_compat_patch():
    import inspect
    import sys
    import importlib

    try:
        from torch.utils._config_typing import Config as _Config
    except Exception:
        # torch not installed yet, or layout differs — nothing to patch.
        return False

    if getattr(_Config, "_llm_scratch_patched", False):
        # Already patched. But we still may need to reload a half-imported
        # torch._dynamo.config if it crashed on first import attempt.
        _reload_dynamo_if_broken(sys, importlib)
        return True

    _orig_init = _Config.__init__

    def _patched_init(self, *args, **kwargs):
        try:
            allowed = _patched_init._allowed_kwargs
        except AttributeError:
            try:
                sig = inspect.signature(_orig_init)
                allowed = set(sig.parameters.keys())
            except (ValueError, TypeError):
                # Couldn't introspect — accept everything (let original raise).
                return _orig_init(self, *args, **kwargs)
            _patched_init._allowed_kwargs = allowed
        clean = {k: v for k, v in kwargs.items() if k in allowed}
        return _orig_init(self, *args, **clean)

    _Config.__init__ = _patched_init
    _Config._llm_scratch_patched = True

    _reload_dynamo_if_broken(sys, importlib)
    return True


def _reload_dynamo_if_broken(sys, importlib):
    """If torch._dynamo.config is in sys.modules but is half-imported (i.e.
    `torch._dynamo.config.skip_code_recursive_on_recompile_limit_hit`
    doesn't exist because the module-level Config() call crashed), reload
    it now that Config.__init__ is patched.
    """
    mod = sys.modules.get("torch._dynamo.config")
    if mod is None:
        return  # not imported yet — will import cleanly next time
    if hasattr(mod, "skip_code_recursive_on_recompile_limit_hit"):
        return  # already imported successfully — nothing to do
    try:
        importlib.reload(mod)
    except Exception:
        # Best effort; if reload fails, the next `import torch._dynamo.config`
        # will retry and either succeed (patch is in place) or raise a clearer
        # error.
        pass


_apply_torch_dynamo_compat_patch()


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
    # Make sure the dynamo compat patch is applied (in case torch got
    # installed after llm_scratch was first imported).
    _apply_torch_dynamo_compat_patch()

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
