"""
Dataset & tokenizer setup.

We default to WikiText-103 + the GPT-2 tokenizer, but you can also plug in a
**custom dataset** via `load_custom_dataset()`:
  - local `.txt`       : one document per line (or per file)
  - local `.json`/`.jsonl`: a "text" field per record (override with `text_field`)
  - local `.csv`       : a "text" column (override with `text_field`)
  - HF Hub dataset name: e.g. "imdb", "wikitext", "roneneldan/TinyStories"
  - URL                : http(s) link to a .txt file

数据集与 tokenizer。默认用 WikiText-103 + GPT-2 tokenizer；也支持通过
`load_custom_dataset()` 加载自定义数据集（本地 .txt / .json / .jsonl / .csv、
HuggingFace Hub 数据集名、或 http(s) URL）。
"""

from __future__ import annotations
import os
import json
import csv
from typing import Any, Dict, Optional, Union

from datasets import load_dataset, Dataset, DatasetDict
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


# ---------------------------------------------------------------------------
# Internal: tokenize + chunk a raw-text dataset into fixed block_size rows
# ---------------------------------------------------------------------------
def _tokenize_and_chunk(
    raw: Dict[str, Dataset],
    tokenizer,
    block_size: int,
    text_field: str,
    num_proc: Optional[int],
    subset_size: Optional[int],
):
    """Shared tokenize+chunk pipeline used by both wikitext and custom loaders.
    wikitext 与自定义加载共用的 tokenize + chunk 流水线。"""

    def cap(ds):
        if subset_size is not None and len(ds) > subset_size:
            ds = ds.select(range(subset_size))
        return ds

    if subset_size is not None:
        raw = {k: cap(v) for k, v in raw.items()}

    # 1) tokenize per-text
    def tokenize_fn(examples):
        return tokenizer(examples[text_field], return_attention_mask=False)

    # Determine which column to drop (the original text_field column).
    cols_to_drop = [c for c in raw["train"].column_names if c != text_field]
    # Drop everything except text_field; map will then drop text_field itself
    # (because of remove_columns=text_field).
    if cols_to_drop:
        raw = {k: v.remove_columns(cols_to_drop) for k, v in raw.items()}

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=[text_field] if text_field in raw["train"].column_names else None,
        desc="Tokenizing",
    )

    # 2) concat all texts and chunk into block_size
    def chunk_fn(examples, _block_size):
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
    # `map` returns DatasetDict (or dict, depending on input type)
    return {k: v for k, v in chunked.items()}


# ---------------------------------------------------------------------------
# WikiText-103 loader
# ---------------------------------------------------------------------------
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
    return _tokenize_and_chunk(
        raw, tokenizer, block_size,
        text_field="text",
        num_proc=num_proc,
        subset_size=subset_size,
    )


# ---------------------------------------------------------------------------
# Custom dataset loader
# ---------------------------------------------------------------------------
def _is_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))


def _looks_like_hf_hub_name(s: str) -> bool:
    """Heuristic: a HF Hub dataset name has no path separators and no file
    extension (or only a config after a comma), and isn't a URL.
    启发式判断：HF Hub 数据集名通常无路径分隔符、无文件后缀（',' 后为 config）。"""
    if not isinstance(s, str) or _is_url(s):
        return False
    if os.path.exists(s):
        return False
    # Has a file extension → probably a file path, not a HF name.
    _, ext = os.path.splitext(s)
    if ext.lower() in (".txt", ".json", ".jsonl", ".csv", ".tsv"):
        return False
    return True


def _load_local_text_file(path: str) -> Dict[str, Dataset]:
    """Load a single .txt file: split into lines, use 90/5/5 train/val/test.
    单个 .txt 文件：按行切分，按 90/5/5 划分 train/val/test。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"File {path!r} is empty or has no non-blank lines.")
    n = len(lines)
    n_train = max(1, int(n * 0.9))
    n_val = max(1, int(n * 0.05))
    train = Dataset.from_dict({"text": lines[:n_train]})
    val = Dataset.from_dict({"text": lines[n_train : n_train + n_val]})
    test = Dataset.from_dict({"text": lines[n_train + n_val :]})
    return {"train": train, "validation": val, "test": test}


def _load_local_json_or_jsonl(path: str, text_field: str) -> Dict[str, Dataset]:
    """Load a .json (list of records) or .jsonl file.
    加载 .json（记录列表）或 .jsonl 文件。"""
    records = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    records.append(json.loads(ln))
    else:  # .json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Maybe a DatasetDict-like {"train": [...], "validation": [...], ...}
            out = {}
            for split, recs in data.items():
                if not isinstance(recs, list):
                    raise ValueError(
                        f"Top-level JSON object key {split!r} must map to a list."
                    )
                texts = [r[text_field] if isinstance(r, dict) else str(r) for r in recs]
                out[split] = Dataset.from_dict({text_field: texts})
            # Ensure required splits exist.
            for s in ("train", "validation", "test"):
                out.setdefault(s, out.get("train"))
            return out
        if not isinstance(data, list):
            raise ValueError(
                f"Expected JSON list or {{split: [records]}} object in {path!r}."
            )
        records = data

    if not records:
        raise ValueError(f"No records found in {path!r}.")
    texts = [r[text_field] if isinstance(r, dict) else str(r) for r in records]
    n = len(texts)
    n_train = max(1, int(n * 0.9))
    n_val = max(1, int(n * 0.05))
    train = Dataset.from_dict({text_field: texts[:n_train]})
    val = Dataset.from_dict({text_field: texts[n_train : n_train + n_val]})
    test = Dataset.from_dict({text_field: texts[n_train + n_val :]})
    return {"train": train, "validation": val, "test": test}


def _load_local_csv(path: str, text_field: str) -> Dict[str, Dataset]:
    """Load a .csv file. If `text_field` column doesn't exist, use the first
    string column. Split 90/5/5 train/val/test.
    加载 .csv 文件：若 `text_field` 列不存在，则用第一个字符串列。"""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV {path!r} is empty.")
    fieldnames = reader.fieldnames or []
    col = text_field if text_field in fieldnames else fieldnames[0]
    texts = [row.get(col, "") or "" for row in rows]
    n = len(texts)
    n_train = max(1, int(n * 0.9))
    n_val = max(1, int(n * 0.05))
    train = Dataset.from_dict({text_field: texts[:n_train]})
    val = Dataset.from_dict({text_field: texts[n_train : n_train + n_val]})
    test = Dataset.from_dict({text_field: texts[n_train + n_val :]})
    return {"train": train, "validation": val, "test": test}


def load_custom_dataset(
    source: Union[str, Dict[str, Dataset]],
    tokenizer,
    block_size: int = 256,
    text_field: str = "text",
    num_proc: Optional[int] = None,
    subset_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    **hf_kwargs,
) -> Dict[str, "datasets.Dataset"]:
    """Load a custom dataset, tokenize + chunk into fixed block_size rows.

    Args:
        source:
            - Path to a local `.txt`, `.json`, `.jsonl`, `.csv`, or `.tsv` file.
            - HuggingFace Hub dataset name (e.g. "imdb", "roneneldan/TinyStories",
              "wikitext", "Anthropic/hh-rlhf"). Loaded via `datasets.load_dataset`.
            - URL (http/https) to a `.txt` file — downloaded by HF `load_dataset`.
            - An already-built dict of {split: datasets.Dataset} — passed through.
        tokenizer: a HF tokenizer with .pad_token set.
        block_size: model context length.
        text_field: name of the column that holds the raw text (default "text").
            For HF Hub datasets that use a different column (e.g. IMDb uses
            "text", TinyStories uses "text", OpenWebText uses "text"), set this
            accordingly.
        num_proc: multiprocessing workers for tokenization.
        subset_size: cap #rows per split for fast smoke tests (None = full).
        cache_dir: HF datasets cache.
        **hf_kwargs: extra kwargs forwarded to `datasets.load_dataset` (e.g.
            `split=`, `data_files=`, `name=`, `revision=`).

    Returns:
        {"train": Dataset, "validation": Dataset, "test": Dataset}
        Each row has fields: input_ids (list[int]), labels (list[int]),
        attention_mask (list[int]). Lengths == block_size.

    加载自定义数据集：
      - 本地 `.txt` / `.json` / `.jsonl` / `.csv` / `.tsv` 文件路径
      - HuggingFace Hub 数据集名（如 "imdb"、"roneneldan/TinyStories"）
      - URL（http/https）指向 `.txt` 文件
      - 已构造好的 {split: datasets.Dataset} 字典（直接透传）
    """
    # --- Already-built dict ------------------------------------------------
    if isinstance(source, dict):
        return _tokenize_and_chunk(
            source, tokenizer, block_size,
            text_field=text_field,
            num_proc=num_proc,
            subset_size=subset_size,
        )

    if not isinstance(source, str) or not source:
        raise ValueError(f"`source` must be a path / URL / HF name / dict, got {source!r}")

    # --- Local file --------------------------------------------------------
    if os.path.exists(source):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".txt":
            raw = _load_local_text_file(source)
            raw = {k: v for k, v in raw.items()}  # already dicts of Dataset
        elif ext in (".json", ".jsonl"):
            raw = _load_local_json_or_jsonl(source, text_field=text_field)
        elif ext in (".csv", ".tsv"):
            raw = _load_local_csv(source, text_field=text_field)
        else:
            raise ValueError(
                f"Unsupported file extension {ext!r} for {source!r}. "
                f"Supported: .txt, .json, .jsonl, .csv, .tsv"
            )
        return _tokenize_and_chunk(
            raw, tokenizer, block_size,
            text_field=text_field,
            num_proc=num_proc,
            subset_size=subset_size,
        )

    # --- URL (let HF datasets handle the download) -------------------------
    if _is_url(source):
        # datasets.load_dataset can fetch a remote text file directly.
        raw = load_dataset("text", data_files=source, cache_dir=cache_dir, **hf_kwargs)
        # Make sure all 3 splits exist; if only "train" came back, slice it.
        if "validation" not in raw and "test" not in raw:
            split = raw["train"].train_test_split(test_size=0.1, seed=42)
            raw = DatasetDict({
                "train": split["train"],
                "validation": split["test"],
                "test": split["test"],
            })
        elif "validation" not in raw:
            raw = DatasetDict({**raw, "validation": raw["test"]})
        return _tokenize_and_chunk(
            raw, tokenizer, block_size,
            text_field=text_field,
            num_proc=num_proc,
            subset_size=subset_size,
        )

    # --- HuggingFace Hub dataset name --------------------------------------
    if _looks_like_hf_hub_name(source):
        raw = load_dataset(source, cache_dir=cache_dir, **hf_kwargs)
        # Ensure required splits exist; if only "train" came back, slice it.
        if "validation" not in raw and "test" not in raw:
            split = raw["train"].train_test_split(test_size=0.1, seed=42)
            raw = DatasetDict({
                "train": split["train"],
                "validation": split["test"],
                "test": split["test"],
            })
        elif "validation" not in raw:
            raw = DatasetDict({**raw, "validation": raw.get("test", raw["train"])})
        # Verify text_field exists in the train split.
        if text_field not in raw["train"].column_names:
            avail = raw["train"].column_names
            raise ValueError(
                f"text_field {text_field!r} not found in dataset {source!r}. "
                f"Available columns: {avail}. Pass `text_field=<col>` to "
                f"load_custom_dataset()."
            )
        return _tokenize_and_chunk(
            raw, tokenizer, block_size,
            text_field=text_field,
            num_proc=num_proc,
            subset_size=subset_size,
        )

    raise ValueError(
        f"Could not resolve dataset source {source!r}. "
        f"Expected a local file path, a URL, a HF Hub dataset name, or a dict."
    )
