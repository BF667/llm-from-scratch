<div align="center">

# 🚀 llm-from-scratch · Latih LLM dari nol

**Latih LLM decoder-only berukuran kecil dari nol — berjalan di Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667/llm-from-scratch/blob/main/notebooks/train_llm_colab.ipynb)
&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
&nbsp;&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)

</div>

---

[English](./readme-en.md) | [简体中文](./readme-zh.md) | [Bahasa Indonesia](./readme-id.md)

---

Proyek sederhana yang **siap dijalankan di Google Colab** untuk melatih LLM
decoder-only berukuran kecil **dari nol** (tanpa bobot pretrained) di
**WikiText-103**, dengan **ukuran model yang bisa dipilih** (`tiny` / `0.1b` /
`0.5b` / `1b` / `custom`) dan **arsitektur yang bisa dipilih** (gaya `gpt2`
atau gaya `llama`). Dibangun di atas HuggingFace `transformers` + `Trainer`.

---

## ✨ Fitur

- ✅ Dua arsitektur, keduanya ditulis dari nol di atas HF Transformers:
  - **Gaya GPT-2**: LayerNorm + posisi absolut yang dipelajari + MLP GELU
  - **Gaya LLaMA**: RMSNorm + RoPE + SwiGLU, tanpa bias, embedding tidak di-tie
- ✅ Lima preset ukuran: `tiny` (~16M), `0.1b` (~95M), `0.5b` (~500M), `1b` (~750M), `custom`
- ✅ Dataset WikiText-103 + tokenizer GPT-2 BPE (vocab=50257)
- ✅ **Dataset kustom** — file lokal `.txt` / `.json` / `.jsonl` / `.csv`, nama HF Hub, atau URL
- ✅ **Lanjutkan training** dari checkpoint mana pun (`--resume_from_checkpoint auto`)
- ✅ Loop training berbasis HF `Trainer` dengan LR cosine, warmup, mixed precision
- ✅ Dukungan gradient checkpointing untuk GPU dengan VRAM terbatas
- ✅ Notebook Colab sekali klik (`notebooks/train_llm_colab.ipynb`)
- ✅ Dokumentasi tiga bahasa (Inggris, Mandarin, Indonesia)

---

## 📁 Struktur proyek

```
llm-from-scratch/
├── README.md                       # pemilih bahasa / language picker
├── docs/                           # dokumentasi lengkap per bahasa
│   ├── readme-en.md
│   ├── readme-zh.md
│   └── readme-id.md
├── requirements.txt                # dependensi Python
├── configs/                        # preset YAML (untuk referensi)
│   ├── tiny.yaml
│   ├── 0.1b.yaml
│   ├── 0.5b.yaml
│   ├── 1b.yaml
│   └── custom.yaml
├── llm_scratch/                    # paket Python
│   ├── __init__.py
│   ├── config.py                   # preset ukuran + pembuat konfigurasi
│   ├── model_gpt2.py               # model gaya GPT-2
│   ├── model_llama.py              # model gaya LLaMA
│   ├── dataset.py                  # WikiText-103 + custom datasets + tokenizer
│   ├── train.py                    # entry point training
│   └── generate.py                 # skrip generasi
└── notebooks/
    └── train_llm_colab.ipynb       # notebook Colab sekali klik
```

---

## 🧮 Preset ukuran

| `size` | n_layer | n_head | n_embd | block_size | ~params | batch×accum default | Colab rekomendasi |
|--------|---------|--------|--------|------------|---------|----------------------|--------------------|
| `tiny`    | 4  | 4  | 256  | 256  | ~16M  | 32 × 1  | T4 gratis ✅ |
| `0.1b`    | 8  | 12 | 768  | 512  | ~95M  | 12 × 2  | T4 gratis (lambat) / Pro ✅ |
| `0.5b`    | 24 | 16 | 1280 | 1024 | ~500M | 4 × 8   | Pro / A100 ✅ |
| `1b`      | 24 | 16 | 1536 | 1024 | ~750M | 2 × 16  | A100 saja ⚠️ |
| `custom`  | atur sendiri |    |      |      |       | 16 × 2  | tergantung |

Jumlah parameter diestimasi untuk `vocab_size=50257`; model gaya LLaMA sedikit
lebih kecil per layer karena pembulatan SwiGLU. Untuk `0.5b` ke atas, aktifkan
`grad_checkpoint=True` agar muat di VRAM yang terbatas.

---

## 🚀 Mulai cepat di Colab

1. Buka [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → T4 GPU** (atau A100 untuk pengguna Pro).
3. Upload folder proyek ini ke Colab — cara termudah: zip dulu di lokal, drag
   zip-nya ke file browser Colab, lalu `!unzip`.
4. Buka `notebooks/train_llm_colab.ipynb` di Colab (**File → Upload notebook**).
5. Jalankan cell dari atas ke bawah. **Di bagian 4, atur `ARCH` dan `SIZE`** —
   ini adalah langkah "memilih parameter".

### Uji asap (T4 gratis, ~10 menit)

```python
ARCH = "gpt2"
SIZE = "tiny"
TRAIN = {"epochs": 1, "subset_size": 2000}
```

### Training serius

```python
ARCH = "llama"
SIZE = "0.1b"
TRAIN = {"epochs": 1}  # WikiText-103 penuh
```

---

## 💻 Penggunaan lokal

### Instalasi

```bash
git clone https://github.com/BF667/llm-from-scratch.git
cd llm-from-scratch
pip install -r requirements.txt
```

### Training

```bash
# GPT-2 tiny + 2000 baris subset untuk uji asap cepat
python -m llm_scratch.train --arch gpt2 --size tiny --output_dir out/gpt2-tiny \
    --subset_size 2000 --epochs 1

# LLaMA 0.1B di WikiText-103 penuh
python -m llm_scratch.train --arch llama --size 0.1b --output_dir out/llama-0.1b

# Ukuran custom — timpa parameter individual
python -m llm_scratch.train --arch gpt2 --size custom \
    --n_layer 12 --n_head 12 --n_embd 768 --block_size 768 \
    --output_dir out/gpt2-custom --grad_checkpoint

# Matikan mixed precision
python -m llm_scratch.train --arch gpt2 --size tiny --no_fp16
```

### Lanjutkan training dari checkpoint 🔄

Set `--resume_from_checkpoint` untuk melanjutkan training dari tempat run
sebelumnya berhenti. HF `Trainer` memulihkan bobot model, state optimizer,
scheduler LR, RNG, dan penghitung langkah global — loss melanjutkan dari
tempatnya berhenti, bukan loncat kembali ke ~10.

```bash
# Auto-pick checkpoint terbaru di out/gpt2-tiny dan lanjutkan 2 epoch lagi
python -m llm_scratch.train --arch gpt2 --size tiny \
    --output_dir out/gpt2-tiny --epochs 2 \
    --resume_from_checkpoint auto

# Lanjutkan dari direktori checkpoint spesifik
python -m llm_scratch.train --arch gpt2 --size tiny \
    --output_dir out/gpt2-tiny --epochs 2 \
    --resume_from_checkpoint out/gpt2-tiny/checkpoint-500
```

Di notebook Colab, set `RESUME_FROM = "auto"` di bagian 8 sebagai gantinya.

> ⚠️ Arsitektur & ukuran harus cocok dengan checkpoint — Anda tidak bisa
> melanjutkan checkpoint `gpt2-tiny` dengan `--arch llama` atau `--size 0.1b`.

### Dataset kustom 📚

Tak mau pakai WikiText-103? Lempar `--dataset custom --dataset_source <sumber>`
untuk training di korpus Anda sendiri. Sumber yang didukung:

| Jenis sumber | Contoh |
|--------------|--------|
| File `.txt` lokal (satu dokumen per baris) | `--dataset_source /content/my_data.txt` |
| File `.json` / `.jsonl` lokal (field `text` per record) | `--dataset_source /content/notes.jsonl` |
| File `.csv` lokal (kolom `text`) | `--dataset_source /content/data.csv` |
| Nama dataset HuggingFace Hub | `--dataset_source roneneldan/TinyStories` |
| URL ke file `.txt` | `--dataset_source https://example.com/data.txt` |

Bila kolom teks bukan `text`, pakai `--text_field <kolom>`
(mis. `--text_field content`).

```bash
# Training di file .txt lokal
python -m llm_scratch.train --arch gpt2 --size tiny \
    --output_dir out/gpt2-custom-data --epochs 1 \
    --dataset custom --dataset_source /content/my_data.txt

# Training di TinyStories dari HF Hub
python -m llm_scratch.train --arch gpt2 --size 0.1b \
    --output_dir out/gpt2-tinystories --epochs 1 \
    --dataset custom --dataset_source roneneldan/TinyStories
```

Di notebook Colab, set `DATASET = "custom"` dan `DATASET_SOURCE = "..."` di
bagian 4. Dataset file lokal tanpa split akan otomatis dipecah 90/5/5.

### Memperbaiki error `Config() got an unexpected keyword argument 'deprecated'` 🔧

Beberapa runtime Colab membawa `torch` yang internal `_dynamo/config.py`-nya
tidak sinkron dengan `_utils/_config_typing.py`, sehingga `import torch`
crash. **Bagian 0** notebook mendeteksi ini dan memasang ulang `torch` /
`torchvision` / `torchaudio` untuk Anda — lalu restart runtime dan jalankan
ulang.

Setara di CLI:

```bash
pip install -U --force-reinstall torch torchvision torchaudio
```

Modul `llm_scratch.config` murni stdlib, jadi
`from llm_scratch.config import build_config, list_presets` selalu jalan
meski torch sendiri rusak (submodul yang bergantung pada torch dimuat
lambat / lazy saat pertama diakses).

### Generasi

```bash
python -m llm_scratch.generate --checkpoint out/gpt2-tiny \
    --prompt "The future of AI is" --max_new_tokens 80
```

### Sebagai library Python

```python
from llm_scratch import build_model, build_config, list_presets

# tampilkan semua preset
print(list_presets())

# bangun LLaMA 0.5B
model = build_model(arch="llama", size="0.5b")
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ukuran custom
model = build_model(arch="gpt2", size="custom",
                    n_layer=12, n_head=12, n_embd=768, block_size=768)
```

---

## 🧠 Detail arsitektur

### Gaya GPT-2 (`model_gpt2.py`)

| Komponen | Pilihan |
|-----------|--------|
| Norm | LayerNorm (pre-norm) |
| Posisi | Embedding posisi absolut yang dipelajari (`nn.Embedding(block_size, n_embd)`) |
| Attention | Multi-head kausal, QKV menyatu, Flash Attention jika tersedia |
| MLP | Ekspansi 4×, GELU |
| LM head | Di-tie dengan token embeddings |
| Bias | Off secara default (gaya modern) |
| Dropout | Dapat dikonfigurasi (default 0.0) |

### Gaya LLaMA (`model_llama.py`)

| Komponen | Pilihan |
|-----------|--------|
| Norm | RMSNorm (tanpa pengurangan mean, tanpa bias) |
| Posisi | Rotary Positional Embeddings (RoPE) |
| Attention | Multi-head kausal dengan RoPE pada Q/K, Flash Attention |
| MLP | SwiGLU (`w2(silu(w1(x)) * w3(x))`), hidden dibulatkan ke `multiple_of=256` |
| LM head | Tidak di-tie dengan embeddings (`tie_word_embeddings=False`) |
| Bias | Off di semua tempat |
| Dropout | Default 0.0 |

Kedua model terdaftar sebagai subclass `PreTrainedModel`, jadi mereka mendukung
`.save_pretrained()`, `.from_pretrained()`, gradient checkpointing, mixed
precision, dan HF `Trainer` secara langsung.

---

## ⚙️ Default training

| Parameter | Default | Catatan |
|------|---------|-------|
| Optimizer | `adamw_torch` | AdamW dengan weight decay terpisah |
| LR | `3e-4` | Konstan per ukuran; decay cosine ke 0 |
| Warmup | 5% dari total langkah | Warmup linear |
| Weight decay | `0.1` | Diterapkan ke semua parameter non-norm / non-bias |
| Skedul LR | Cosine | Decay ke 0 di akhir training |
| Mixed precision | bf16 jika didukung, jika tidak fp16 | Dimatikan jika tidak ada CUDA |
| Gradient checkpointing | Off | Aktifkan dengan `--grad_checkpoint` atau `model.gradient_checkpointing_enable()` |
| Eval | setiap 500 langkah | Pada split `validation` |
| Simpan | setiap 500 langkah, 2 terakhir disimpan | |

---

## 📊 Hasil yang diharapkan

Di **WikiText-103**, training selama **1 epoch**:

| size | arch | train loss | val loss | ppl | waktu di T4 | waktu di A100 |
|------|------|-----------|----------|-----|------------|--------------|
| tiny | gpt2 | ~4.5 | ~4.8 | ~120 | ~10 menit | ~3 menit |
| 0.1b | gpt2 | ~3.4 | ~3.7 | ~40 | ~1.5 jam | ~20 menit |
| 0.1b | llama | ~3.3 | ~3.6 | ~36 | ~2 jam | ~25 menit |
| 0.5b | llama | ~2.9 | ~3.2 | ~24 | OOM di T4 | ~2 jam |

Ini angka perkiraan kasar — hasil Anda akan bervariasi tergantung `block_size`,
`batch_size`, `grad_accum`, dan beban Colab. Jangan berharap bahasa Inggris
yang koheren dari `tiny`; untuk generasi yang benar-benar dapat dibaca,
setidaknya butuh `0.1b`.

---

## 🧪 Tips & peringatan

- **Run pertama mengunduh WikiText-103** (~180MB). Run berikutnya menggunakan cache.
- **Colab free T4 memiliki VRAM 16GB.** `0.5b` akan OOM tanpa gradient
  checkpointing + batch kecil + grad accum. `1b` pada dasarnya hanya demo di
  Colab — gunakan A100 atau sewa GPU cloud.
- **Untuk training LLM from-scratch yang serius**, proyek ini adalah kerangka
  pengajaran. Untuk kualitas yang baik Anda biasanya butuh: lebih banyak data
  (mis. FineWeb-Edu), konteks lebih panjang, lebih banyak langkah, restart
  skedul cosine, dan vocab tokenizer yang lebih besar (mis. 32k BPE yang
  dilatih di korpus Anda). Repo ini memberi kerangkanya; tukar bagian sesuai kebutuhan.
- **Reproduksibilitas**: set `--seed` (belum diekspos — tambahkan ke
  `TrainingArguments` jika diperlukan).
- **Multi-GPU**: konfigurasi `Trainer` saat ini mendukung single-GPU. Untuk DDP,
  bungkus dengan `accelerate launch`.

---

## 📜 Lisensi

MIT — lakukan apa saja yang Anda mau, jangan salahkan saya jika LLM Anda berhalusinasi.
