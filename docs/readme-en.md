# 🚀 llm-from-scratch · Train a small LLM from scratch

[English](./readme-en.md) | [简体中文](./readme-zh.md) | [Bahasa Indonesia](./readme-id.md)

---

A simple, **Google-Colab-ready** project to train a small decoder-only LLM
**from scratch** (no pretrained weights) on **WikiText-103**, with
**selectable model size** (`tiny` / `0.1b` / `0.5b` / `1b` / `custom`) and
**selectable architecture** (`gpt2` style or `llama` style). Built on top of
HuggingFace `transformers` + `Trainer`.

---

## ✨ Features

- ✅ Two architectures, both written from scratch on HF Transformers:
  - **GPT-2 style**: LayerNorm + learned absolute positions + GELU MLP
  - **LLaMA style**: RMSNorm + RoPE + SwiGLU, no bias, untied embeddings
- ✅ Five size presets: `tiny` (~16M), `0.1b` (~95M), `0.5b` (~500M), `1b` (~750M), `custom`
- ✅ WikiText-103 dataset, GPT-2 BPE tokenizer (vocab=50257)
- ✅ HF `Trainer`-based training loop with cosine LR schedule, warmup, mixed precision
- ✅ Gradient checkpointing support for memory-constrained GPUs
- ✅ One-click Colab notebook (`notebooks/train_llm_colab.ipynb`)
- ✅ Trilingual docs (English, Chinese, Indonesian)

---

## 📁 Project layout

```
llm-from-scratch/
├── README.md                       # language picker
├── docs/                           # full docs per language
│   ├── readme-en.md
│   ├── readme-zh.md
│   └── readme-id.md
├── requirements.txt                # python deps
├── configs/                        # yaml presets (for reference)
│   ├── tiny.yaml
│   ├── 0.1b.yaml
│   ├── 0.5b.yaml
│   ├── 1b.yaml
│   └── custom.yaml
├── llm_scratch/                    # the python package
│   ├── __init__.py
│   ├── config.py                   # size presets + config builder
│   ├── model_gpt2.py               # GPT-2 style model
│   ├── model_llama.py              # LLaMA style model
│   ├── dataset.py                  # WikiText-103 + tokenizer
│   ├── train.py                    # training entry
│   └── generate.py                 # generation script
└── notebooks/
    └── train_llm_colab.ipynb       # one-click Colab notebook
```

---

## 🧮 Size presets

| `size` | n_layer | n_head | n_embd | block_size | ~params | default batch×accum | recommended Colab |
|--------|---------|--------|--------|------------|---------|----------------------|--------------------|
| `tiny`    | 4  | 4  | 256  | 256  | ~16M  | 32 × 1  | free T4 ✅ |
| `0.1b`    | 8  | 12 | 768  | 512  | ~95M  | 12 × 2  | free T4 (slow) / Pro ✅ |
| `0.5b`    | 24 | 16 | 1280 | 1024 | ~500M | 4 × 8   | Pro / A100 ✅ |
| `1b`      | 24 | 16 | 1536 | 1024 | ~750M | 2 × 16  | A100 only ⚠️ |
| `custom`  | you set |    |      |      |       | 16 × 2  | depends |

Param counts are estimated for `vocab_size=50257`; LLaMA-style models are
slightly smaller per layer due to the SwiGLU rounding. For `0.5b` and above,
set `grad_checkpoint=True` to fit in limited VRAM.

---

## 🚀 Quick start on Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → T4 GPU** (or A100 on Pro).
3. Upload this whole folder to Colab — easiest way: zip it locally, drag the
   zip into the Colab file browser, then `!unzip`.
4. Open `notebooks/train_llm_colab.ipynb` in Colab (**File → Upload notebook**).
5. Run cells top to bottom. **In section 4, set `ARCH` and `SIZE`** — this is
   the "select parameters" step.

### Smoke test (free T4, ~10 minutes)

```python
ARCH = "gpt2"
SIZE = "tiny"
TRAIN = {"epochs": 1, "subset_size": 2000}
```

### Real training

```python
ARCH = "llama"
SIZE = "0.1b"
TRAIN = {"epochs": 1}  # full WikiText-103
```

---

## 💻 Local usage

### Install

```bash
git clone https://github.com/BF667/llm-from-scratch.git
cd llm-from-scratch
pip install -r requirements.txt
```

### Train

```bash
# Tiny GPT-2 on a 2000-row subset for a quick smoke test
python -m llm_scratch.train --arch gpt2 --size tiny --output_dir out/gpt2-tiny \
    --subset_size 2000 --epochs 1

# LLaMA 0.1B on the full WikiText-103
python -m llm_scratch.train --arch llama --size 0.1b --output_dir out/llama-0.1b

# Custom size — override individual knobs
python -m llm_scratch.train --arch gpt2 --size custom \
    --n_layer 12 --n_head 12 --n_embd 768 --block_size 768 \
    --output_dir out/gpt2-custom --grad_checkpoint

# Disable mixed precision
python -m llm_scratch.train --arch gpt2 --size tiny --no_fp16
```

### Generate

```bash
python -m llm_scratch.generate --checkpoint out/gpt2-tiny \
    --prompt "The future of AI is" --max_new_tokens 80
```

### Use as a Python library

```python
from llm_scratch import build_model, build_config, list_presets

# show all presets
print(list_presets())

# build a 0.5B LLaMA
model = build_model(arch="llama", size="0.5b")
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# custom size
model = build_model(arch="gpt2", size="custom",
                    n_layer=12, n_head=12, n_embd=768, block_size=768)
```

---

## 🧠 Architecture details

### GPT-2 style (`model_gpt2.py`)

| Component | Choice |
|-----------|--------|
| Norm | LayerNorm (pre-norm) |
| Position | Learned absolute embeddings (`nn.Embedding(block_size, n_embd)`) |
| Attention | Multi-head causal, fused QKV, Flash Attention when available |
| MLP | 4× expansion, GELU |
| LM head | Tied with token embeddings |
| Bias | Off by default (modern style) |
| Dropout | Configurable (0.0 by default) |

### LLaMA style (`model_llama.py`)

| Component | Choice |
|-----------|--------|
| Norm | RMSNorm (no mean subtraction, no bias) |
| Position | Rotary Positional Embeddings (RoPE) |
| Attention | Multi-head causal with RoPE applied to Q/K, Flash Attention |
| MLP | SwiGLU (`w2(silu(w1(x)) * w3(x))`), hidden rounded to `multiple_of=256` |
| LM head | Untied from embeddings (`tie_word_embeddings=False`) |
| Bias | Off everywhere |
| Dropout | 0.0 by default |

Both models are registered as `PreTrainedModel` subclasses, so they support
`.save_pretrained()`, `.from_pretrained()`, gradient checkpointing, mixed
precision, and HF `Trainer` out of the box.

---

## ⚙️ Training defaults

| Knob | Default | Notes |
|------|---------|-------|
| Optimizer | `adamw_torch` | AdamW with decoupled weight decay |
| LR | `3e-4` | Per-size constant; cosine decay to 0 |
| Warmup | 5% of total steps | Linear warmup |
| Weight decay | `0.1` | Applied to all non-norm/non-bias params |
| LR schedule | Cosine | Decays to 0 by end of training |
| Mixed precision | bf16 if supported, else fp16 | Disabled if no CUDA |
| Gradient checkpointing | Off | Enable with `--grad_checkpoint` or `model.gradient_checkpointing_enable()` |
| Eval | every 500 steps | On the `validation` split |
| Save | every 500 steps, last 2 kept | |

---

## 📊 Expected results

On **WikiText-103**, training for **1 epoch**:

| size | arch | train loss | val loss | ppl | time on T4 | time on A100 |
|------|------|-----------|----------|-----|------------|--------------|
| tiny | gpt2 | ~4.5 | ~4.8 | ~120 | ~10 min | ~3 min |
| 0.1b | gpt2 | ~3.4 | ~3.7 | ~40 | ~1.5 h | ~20 min |
| 0.1b | llama | ~3.3 | ~3.6 | ~36 | ~2 h | ~25 min |
| 0.5b | llama | ~2.9 | ~3.2 | ~24 | OOM on T4 | ~2 h |

These are rough ballpark numbers — your mileage will vary with `block_size`,
`batch_size`, `grad_accum`, and Colab load. Don't expect coherent English from
`tiny`; for actual readable generations you want at least `0.1b`.

---

## 🧪 Tips & caveats

- **First run downloads WikiText-103** (~180MB). Subsequent runs use the cache.
- **Colab free T4 has 16GB VRAM.** `0.5b` will OOM without gradient
  checkpointing + small batch + grad accum. `1b` is essentially demo-only on
  Colab — use A100 or rent a cloud GPU.
- **For real from-scratch LLM training**, this project is a teaching scaffold.
  To get good quality you typically need: more data (e.g. FineWeb-Edu),
  longer context, more steps, cosine schedule restarts, and a larger tokenizer
  vocab (e.g. 32k BPE trained on your corpus). This repo gives you the
  skeleton; swap parts in as needed.
- **Reproducibility**: set `--seed` (not exposed yet — add it to
  `TrainingArguments` if you need it).
- **Multi-GPU**: the current `Trainer` config supports single-GPU. For DDP,
  wrap with `accelerate launch`.

---

## 📜 License

MIT — do whatever you want, just don't blame me if your LLM hallucinates.
