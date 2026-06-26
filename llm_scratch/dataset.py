"""
Dataset & tokenizer setup — WikiText-103 with the GPT-2 tokenizer.
数据集与 tokenizer —— 使用 WikiText-103 + GPT-2 tokenizer。

We use the GPT-2 BPE tokenizer so vocab_size=50257 works out of the box
and we don't have to train our own tokenizer (saves hours on Colab).
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_TOKENIZER = "gpt2"
WIKITEXT_RAW_BLOCK = 1024   # chunk size before grouping into block_size
WIKITEXT_VARIANT = "wikitext-103-raw-v1"


def build_tokenizer(tokenizer_name: str = DEFAULT_TOKENIZER):
    """Load a HF tokenizer; pad token = eos token for GPT-2 family.
    加载 HF tokenizer；GPT-2 系列无 pad token，用 eos 代替。"""
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_wikitext_dataset(
    tokenizer,
    block_size: int = 256,
    streaming: bool = False,
    num_proc: Optional[int] = None,
    subset_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, "datasets.Dataset"]:
    """Load WikiText-103 and tokenize + chunk into fixed block_size.

    Args:
        tokenizer: a HF tokenizer with .pad_token set
        block_size: model context length
        streaming: if True, return iterable datasets (low memory)
        num_proc: multiprocessing workers for tokenization
        subset_size: cap #rows per split for fast smoke tests (None = full)
        cache_dir: HF datasets cache

    Returns:
        {"train": Dataset, "validation": Dataset, "test": Dataset}
        Each row has fields: input_ids (list[int]), labels (list[int]),
        attention_mask (list[int]). Lengths == block_size.
    """
    raw = load_dataset(WIKITEXT_VARIANT, cache_dir=cache_dir)

    def cap(ds):
        if subset_size is not None and len(ds) > subset_size:
            ds = ds.select(range(subset_size))
        return ds

    if subset_size is not None:
        raw = {k: cap(v) for k, v in raw.items()}

    # 1) tokenize per-text
    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=False)

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    # 2) concat all texts and chunk into block_size
    def chunk_fn(examples, _block_size):
        # examples: {"input_ids": [[...], [...], ...]}
        all_ids = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)  # EOS between docs
        chunks = []
        n = (len(all_ids) // _block_size) * _block_size
        for i in range(0, n, _block_size):
            chunk = all_ids[i : i + _block_size]
            chunks.append({
                "input_ids": chunk,
                "labels": list(chunk),                      # shift done inside model
                "attention_mask": [1] * _block_size,
            })
        return {"input_ids": [c["input_ids"] for c in chunks],
                "labels": [c["labels"] for c in chunks],
                "attention_mask": [c["attention_mask"] for c in chunks]}

    chunked = tokenized.map(
        chunk_fn,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"_block_size": block_size},
        desc="Chunking",
    )
    # `map` returns DatasetDict
    return {k: v for k, v in chunked.items()}
