<div align="center">

# 🚀 llm-from-scratch

**Train a small decoder-only LLM from scratch — runs on Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667/llm-from-scratch/blob/main/notebooks/train_llm_colab.ipynb)
&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
&nbsp;&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)

Selectable architecture (`gpt2` or `llama`) ·
Selectable size (`tiny` / `0.1b` / `0.5b` / `1b` / `custom`) ·
WikiText-103 or custom datasets · Continue training from checkpoints ·
HuggingFace `transformers` + `Trainer`

</div>

---

## 🌐 Documentation / 文档 / Dokumentasi

| Language | Link |
|----------|------|
| 🇬🇧 English | [docs/readme-en.md](./docs/readme-en.md) |
| 🇨🇳 简体中文 | [docs/readme-zh.md](./docs/readme-zh.md) |
| 🇮🇩 Bahasa Indonesia | [docs/readme-id.md](./docs/readme-id.md) |

---

## ⚡ Quick links

- 📓 **Colab notebook** — click the badge above, or open [`notebooks/train_llm_colab.ipynb`](./notebooks/train_llm_colab.ipynb) directly
- 🧠 GPT-2 model: [`llm_scratch/model_gpt2.py`](./llm_scratch/model_gpt2.py)
- 🦙 LLaMA model: [`llm_scratch/model_llama.py`](./llm_scratch/model_llama.py)
- ⚙️ Training entry: [`llm_scratch/train.py`](./llm_scratch/train.py)
- 📦 Size presets: [`llm_scratch/config.py`](./llm_scratch/config.py)
- ⚙️ YAML presets: [`configs/`](./configs/)

---

## 🧮 Size presets at a glance

| `size` | ~params | recommended Colab |
|--------|---------|--------------------|
| `tiny`    | ~16M  | free T4 ✅ |
| `0.1b`    | ~95M  | free T4 (slow) / Pro ✅ |
| `0.5b`    | ~500M | Pro / A100 ✅ |
| `1b`      | ~750M | A100 only ⚠️ |
| `custom`  | you set | depends |

---

## 🚀 One-line smoke test

```bash
python -m llm_scratch.train --arch gpt2 --size tiny --output_dir out/gpt2-tiny \
    --subset_size 2000 --epochs 1
```

➡️ **Full setup, architecture details, training tips, and expected results are
in the per-language docs above.**

---

## 📜 License

MIT
