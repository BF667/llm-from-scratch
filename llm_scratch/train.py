"""
Training entry point using HF Trainer.
基于 HF Trainer 的训练入口。

CLI usage:
    python -m llm_scratch.train --arch gpt2 --size 0.1b --output_dir out/gpt2-0.1b

API usage:
    from llm_scratch import run_training
    run_training(arch="llama", size="0.5b", output_dir="out/llama-0.5b")
"""

from __future__ import annotations
import os
import math
import argparse
import logging
from dataclasses import asdict
from typing import Optional

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from .config import build_config, estimate_params, SIZE_PRESETS
from .model_gpt2 import GPTScratchConfig, GPTScratchForCausalLM
from .model_llama import LlamaScratchConfig, LlamaScratchForCausalLM
from .dataset import (
    build_tokenizer,
    load_wikitext_dataset,
    load_custom_dataset,
)

logger = logging.getLogger("llm_scratch.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model_from_config(cfg: dict):
    """Build a model + matching PretrainedConfig from our dict config.
    从 dict 配置构建模型 + 对应的 PretrainedConfig。"""
    arch = cfg["arch"]
    if arch == "gpt2":
        mcfg = GPTScratchConfig(
            vocab_size=cfg["vocab_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
        )
        model = GPTScratchForCausalLM(mcfg)
    elif arch == "llama":
        mcfg = LlamaScratchConfig(
            vocab_size=cfg["vocab_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
        )
        model = LlamaScratchForCausalLM(mcfg)
    else:
        raise ValueError(f"Unknown arch: {arch!r}")
    return model, mcfg


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _has_checkpoint(output_dir: str) -> bool:
    """Return True if `output_dir` contains at least one HF checkpoint subdir
    (i.e. a dir matching `checkpoint-<step>`).
    判断 output_dir 中是否存在至少一个 HF checkpoint 子目录（形如 `checkpoint-<step>`）。
    """
    import os, re
    if not os.path.isdir(output_dir):
        return False
    pat = re.compile(r"^checkpoint-\d+$")
    return any(pat.match(d) for d in os.listdir(output_dir))


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def run_training(
    arch: str = "gpt2",
    size: str = "tiny",
    output_dir: str = "out",
    epochs: Optional[float] = None,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
    lr: Optional[float] = None,
    subset_size: Optional[int] = None,
    grad_checkpoint: bool = False,
    fp16: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500,
    log_steps: int = 20,
    max_steps: Optional[int] = None,
    num_proc: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    dataset: str = "wikitext",
    dataset_source: Optional[str] = None,
    text_field: str = "text",
    **overrides,
) -> str:
    """Run end-to-end training. Returns the output_dir.

    Train the selected arch + size on WikiText-103 (default) or a custom
    dataset. Returns the output dir.

    Args:
        resume_from_checkpoint:
            - None  → train from scratch (random init).
            - "auto" or True → auto-detect the latest checkpoint inside
              `output_dir` and resume from it (optimizer state, LR schedule,
              step counter, RNG all restored). If no checkpoint exists, falls
              back to from-scratch training with a warning.
            - a path → resume from that specific checkpoint directory.
        dataset:
            - "wikitext" → use WikiText-103 (default).
            - "custom"   → use `dataset_source` (see below).
        dataset_source:
            Required when `dataset="custom"`. Can be:
              - Path to a local `.txt` / `.json` / `.jsonl` / `.csv` / `.tsv` file
              - HuggingFace Hub dataset name (e.g. "imdb",
                "roneneldan/TinyStories")
              - URL (http/https) to a `.txt` file
              - dict of {split: datasets.Dataset}
        text_field:
            Column name that holds the raw text in a custom dataset
            (default "text"; for IMDb/TinyStories this is also "text").

            在 WikiText-103（默认）或自定义数据集上训练所选架构 + 尺寸的模型，
            返回输出目录。
            resume_from_checkpoint=None 从头训练；"auto"/True 自动从 output_dir
            中最新 checkpoint 续训；传入路径则从该 checkpoint 续训。
            dataset="wikitext" 用 WikiText-103；dataset="custom" 用 dataset_source
            指向的数据（本地文件 / HF Hub 名 / URL / dict）。
    """
    # 1) Config
    cfg = build_config(arch=arch, size=size, **overrides)
    t = cfg["training"]
    epochs = epochs if epochs is not None else t["epochs"]
    batch_size = batch_size if batch_size is not None else t["batch_size"]
    grad_accum = grad_accum if grad_accum is not None else t["grad_accum"]
    lr = lr if lr is not None else t["lr"]
    warmup_ratio = t["warmup_ratio"]

    # Normalize resume_from_checkpoint: "auto"/True → "auto" for HF Trainer.
    # HF Trainer accepts True (= auto-find latest in output_dir) or a path str.
    resume_arg: Optional[str] = None
    if resume_from_checkpoint in ("auto", True, "true", "True"):
        resume_arg = True
    elif isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() not in ("none", ""):
        resume_arg = resume_from_checkpoint
    # else None → train from scratch

    n_params = estimate_params(cfg)
    logger.info("=" * 60)
    logger.info(f"Arch:  {arch}")
    logger.info(f"Size:  {size}  (~{n_params/1e6:.1f}M params)")
    logger.info(f"Config: {cfg}")
    logger.info(f"Training: epochs={epochs} bs={batch_size} "
                f"accum={grad_accum} lr={lr} fp16={fp16}")
    if resume_arg is None:
        logger.info("Mode: train from scratch / 从头训练")
    elif resume_arg is True:
        logger.info(f"Mode: auto-resume from latest checkpoint in {output_dir} / "
                    f"自动从 {output_dir} 中最新 checkpoint 续训")
    else:
        logger.info(f"Mode: resume from {resume_arg} / 从 {resume_arg} 续训")
    logger.info("=" * 60)

    # 2) Tokenizer + dataset
    tokenizer = build_tokenizer("gpt2")
    cfg["vocab_size"] = len(tokenizer)  # ensure consistency
    logger.info(f"Loaded tokenizer: {tokenizer.__class__.__name__}, vocab={len(tokenizer)}")

    dataset_choice = (dataset or "wikitext").lower()
    if dataset_choice == "wikitext":
        logger.info("Dataset: WikiText-103 / 使用 WikiText-103")
        datasets = load_wikitext_dataset(
            tokenizer=tokenizer,
            block_size=cfg["block_size"],
            subset_size=subset_size,
            num_proc=num_proc,
        )
    elif dataset_choice == "custom":
        if not dataset_source:
            raise ValueError(
                "dataset='custom' requires `dataset_source` (a local file path, "
                "HF Hub name, URL, or dict of Datasets). / "
                "dataset='custom' 需要传入 dataset_source。"
            )
        logger.info(f"Dataset: custom ({dataset_source!r}) / 自定义数据集")
        datasets = load_custom_dataset(
            source=dataset_source,
            tokenizer=tokenizer,
            block_size=cfg["block_size"],
            text_field=text_field,
            subset_size=subset_size,
            num_proc=num_proc,
        )
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Use 'wikitext' or 'custom'. / "
            f"未知 dataset {dataset!r}，请用 'wikitext' 或 'custom'。"
        )
    logger.info(f"Train rows: {len(datasets['train'])}, "
                f"val rows: {len(datasets['validation'])}, "
                f"test rows: {len(datasets['test'])}")

    # 3) Model
    model, mcfg = build_model_from_config(cfg)
    if grad_checkpoint:
        model.gradient_checkpointing_enable()
    n_actual = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_actual/1e6:.2f}M")

    # 4) Training args
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        logging_steps=log_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        fp16=fp16 and torch.cuda.is_available() and not (torch.cuda.is_bf16_supported()),
        bf16=fp16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        load_best_model_at_end=False,
        dataloader_num_workers=2,
        gradient_checkpointing=grad_checkpoint,
        optim="adamw_torch",
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    # 6) Train (optionally resume from a checkpoint)
    #    HF Trainer restores optimizer state, LR scheduler, RNG, and the global
    #    step counter when resume_from_checkpoint is set. The model weights are
    #    also reloaded from the checkpoint, so `model` constructed above is
    #    effectively overwritten at train start.
    #    HF Trainer 会从 checkpoint 恢复优化器状态、学习率调度、随机数状态与全局步数；
    #    模型权重也会从 checkpoint 重新加载，因此上面构建的 model 会被覆盖。
    if resume_arg is True and not _has_checkpoint(output_dir):
        logger.warning(
            f"--resume_from_checkpoint=auto but no checkpoint found in "
            f"{output_dir!r}. Falling back to from-scratch training. / "
            f"未在 {output_dir!r} 中找到 checkpoint，回退为从头训练。"
        )
        resume_arg = None

    train_result = trainer.train(resume_from_checkpoint=resume_arg)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7) Eval
    eval_metrics = trainer.evaluate()
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    eval_metrics["eval_perplexity"] = perplexity
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"Done. Output dir: {output_dir}")
    logger.info(f"Final eval loss: {eval_metrics['eval_loss']:.4f}, "
                f"perplexity: {perplexity:.2f}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="Train a small LLM from scratch on WikiText-103. "
                    "在 WikiText-103 上从头训练小型 LLM。"
    )
    p.add_argument("--arch", choices=["gpt2", "llama"], default="gpt2",
                   help="Model architecture / 模型架构 (default: gpt2)")
    p.add_argument("--size", choices=list(SIZE_PRESETS.keys()), default="tiny",
                   help="Size preset / 尺寸预设 (default: tiny)")
    p.add_argument("--output_dir", default="out",
                   help="Output directory / 输出目录")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--subset_size", type=int, default=None,
                   help="Cap rows per split for smoke tests / 子集行数上限（冒烟测试用）")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="Enable gradient checkpointing / 开启梯度检查点（省显存）")
    p.add_argument("--no_fp16", action="store_true",
                   help="Disable mixed precision / 关闭混合精度")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None,
                   help="Override n_layer (custom size) / 覆盖层数")
    p.add_argument("--n_head", type=int, default=None,
                   help="Override n_head / 覆盖头数")
    p.add_argument("--n_embd", type=int, default=None,
                   help="Override n_embd / 覆盖嵌入维度")
    p.add_argument("--block_size", type=int, default=None,
                   help="Override block_size / 覆盖上下文长度")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Continue training from a checkpoint. Use 'auto' or "
                        "'true' to auto-pick the latest checkpoint in "
                        "--output_dir, or pass an explicit checkpoint dir path. "
                        "Default: None (train from scratch). / "
                        "从 checkpoint 续训：'auto'/'true' 自动找 output_dir 中"
                        "最新的 checkpoint，或传入 checkpoint 目录路径。"
                        "默认 None（从头训练）。")
    p.add_argument("--dataset", choices=["wikitext", "custom"], default="wikitext",
                   help="Dataset to train on. 'wikitext' = WikiText-103 (default). "
                        "'custom' = use --dataset_source. / "
                        "训练数据集：'wikitext'=WikiText-103（默认），"
                        "'custom'=用 --dataset_source 指定的数据。")
    p.add_argument("--dataset_source", default=None,
                   help="Custom dataset source (only used when --dataset=custom). "
                        "Accepts: local file path (.txt/.json/.jsonl/.csv/.tsv), "
                        "HuggingFace Hub dataset name (e.g. 'imdb', "
                        "'roneneldan/TinyStories'), or URL to a .txt file. / "
                        "自定义数据集来源（仅当 --dataset=custom 时生效）：本地文件路径、"
                        "HF Hub 名、或 .txt 文件的 URL。")
    p.add_argument("--text_field", default="text",
                   help="Column name that holds the raw text in a custom dataset "
                        "(default: 'text'). / "
                        "自定义数据集中存放原始文本的列名（默认 'text'）。")
    return p.parse_args()


def main():
    args = _parse_args()
    overrides = {}
    for k in ("n_layer", "n_head", "n_embd", "block_size"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v

    run_training(
        arch=args.arch,
        size=args.size,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        subset_size=args.subset_size,
        grad_checkpoint=args.grad_checkpoint,
        fp16=not args.no_fp16,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        dataset=args.dataset,
        dataset_source=args.dataset_source,
        text_field=args.text_field,
        **overrides,
    )


if __name__ == "__main__":
    main()
