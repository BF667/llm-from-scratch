"""
Model & training configuration presets.
模型与训练配置预设。

Each preset is a small, Colab-friendly configuration. Override any field
by passing kwargs to build_config().
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Size presets — param counts assume vocab_size=50257 (GPT-2 tokenizer).
# 尺寸预设 — 参数量按 vocab_size=50257 (GPT-2 tokenizer) 估算。
# ---------------------------------------------------------------------------
SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    # ~10M params — quick smoke test, trains in minutes on Colab free T4.
    "tiny": {
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "block_size": 256,
        "description": "~10M params, smoke-test size, runs on Colab free T4",
    },
    # ~100M params — small but produces coherent tokens on WikiText.
    "0.1b": {
        "n_layer": 8,
        "n_head": 12,
        "n_embd": 768,
        "block_size": 512,
        "description": "~100M params, small but coherent on WikiText",
    },
    # ~500M params — needs Colab Pro / A100 for reasonable speed.
    "0.5b": {
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1280,
        "block_size": 1024,
        "description": "~500M params, needs A100 / Colab Pro",
    },
    # ~1B params — demo only on Colab, enable gradient checkpointing.
    "1b": {
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1536,
        "block_size": 1024,
        "description": "~1B params, demo only on Colab (enable grad checkpointing)",
    },
    # Custom — caller supplies n_layer/n_head/n_embd/block_size.
    "custom": {
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "block_size": 512,
        "description": "Custom — override n_layer/n_head/n_embd/block_size freely",
    },
}


# Default training hyperparameters per size (Colab-friendly).
# 每个尺寸默认的训练超参数（适配 Colab）。
TRAIN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "tiny":  dict(batch_size=32, grad_accum=1, lr=3e-4, epochs=1, warmup_ratio=0.05),
    "0.1b":  dict(batch_size=12, grad_accum=2, lr=3e-4, epochs=1, warmup_ratio=0.05),
    "0.5b":  dict(batch_size=4,  grad_accum=8, lr=3e-4, epochs=1, warmup_ratio=0.05),
    "1b":    dict(batch_size=2,  grad_accum=16, lr=3e-4, epochs=1, warmup_ratio=0.05),
    "custom":dict(batch_size=16, grad_accum=2, lr=3e-4, epochs=1, warmup_ratio=0.05),
}


def list_presets() -> Dict[str, Dict[str, Any]]:
    """Return a copy of all size presets with descriptions."""
    return {k: dict(v) for k, v in SIZE_PRESETS.items()}


def build_config(
    arch: str = "gpt2",
    size: str = "tiny",
    vocab_size: int = 50257,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a full config dict from a size preset + overrides.

    Args:
        arch: "gpt2" or "llama"
        size: one of SIZE_PRESETS keys
        vocab_size: tokenizer vocab (GPT-2 = 50257)
        **overrides: any field to override (n_layer, n_head, n_embd, block_size, ...)

    Returns:
        dict with all model + training fields.
    """
    if size not in SIZE_PRESETS:
        raise ValueError(
            f"Unknown size {size!r}. Choices: {list(SIZE_PRESETS)}"
        )
    arch = arch.lower()
    if arch not in ("gpt2", "llama"):
        raise ValueError(f"Unknown arch {arch!r}. Use 'gpt2' or 'llama'.")

    cfg = dict(SIZE_PRESETS[size])
    cfg.pop("description", None)
    cfg.update(
        arch=arch,
        size=size,
        vocab_size=vocab_size,
    )
    # training defaults
    t = dict(TRAIN_DEFAULTS[size])
    cfg["training"] = t

    # apply overrides — split model vs training
    model_keys = {"n_layer", "n_head", "n_embd", "block_size"}
    for k, v in overrides.items():
        if k in model_keys or k in ("vocab_size",):
            cfg[k] = v
        elif k in t:
            cfg["training"][k] = v
        else:
            # Unknown key — accept anyway (forward-compat for new arch knobs).
            cfg[k] = v

    return cfg


def estimate_params(cfg: Dict[str, Any]) -> int:
    """Rough parameter count estimate for an embedding+tied-head decoder.
    对带 tied embedding 的 decoder 做粗略参数量估算。
    """
    V, D, L, H = cfg["vocab_size"], cfg["n_embd"], cfg["n_layer"], cfg["n_head"]
    # embedding (tied with LM head): V*D
    emb = V * D
    # per-layer: roughly 12*D^2 for GPT-2 (qkv + proj + mlp up + down)
    # For LLaMA SwiGLU, MLP is ~ (8/3)*D^2, so per-layer ~ 11*D^2.
    per_layer_gpt2 = 12 * D * D
    per_layer_llama = 11 * D * D
    per_layer = per_layer_gpt2 if cfg["arch"] == "gpt2" else per_layer_llama
    total = emb + L * per_layer
    return int(total)


if __name__ == "__main__":
    # Quick sanity print.
    import json
    for s in SIZE_PRESETS:
        for a in ("gpt2", "llama"):
            c = build_config(arch=a, size=s)
            print(f"{a:5s} {s:7s} -> ~{estimate_params(c)/1e6:6.1f}M params")
