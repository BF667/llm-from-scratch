# 🚀 llm-from-scratch · 从头训练小型 LLM

[English](./readme-en.md) | [简体中文](./readme-zh.md) | [Bahasa Indonesia](./readme-id.md)

---

一个**可在 Google Colab 上运行**的简单项目：在 **WikiText-103** 上**从头**训练
小型 decoder-only LLM（无预训练权重），支持**可选模型尺寸**（`tiny` / `0.1b` /
`0.5b` / `1b` / `custom`）与**可选架构**（`gpt2` 风格或 `llama` 风格）。底层基于
HuggingFace `transformers` + `Trainer`。

---

## ✨ 特性

- ✅ 两种架构，均基于 HF Transformers 从头实现：
  - **GPT-2 风格**：LayerNorm + 学习式绝对位置编码 + GELU MLP
  - **LLaMA 风格**：RMSNorm + RoPE + SwiGLU，无 bias，解绑 embedding
- ✅ 五种尺寸预设：`tiny` (~16M)、`0.1b` (~95M)、`0.5b` (~500M)、`1b` (~750M)、`custom`
- ✅ WikiText-103 数据集 + GPT-2 BPE tokenizer（词表 50257）
- ✅ 基于 HF `Trainer` 的训练循环，支持 cosine 学习率、warmup、混合精度
- ✅ 梯度检查点支持，适配显存受限的 GPU
- ✅ 一键 Colab 笔记本（`notebooks/train_llm_colab.ipynb`）
- ✅ 中英双语文档（代码注释英文 + 说明中文）

---

## 📁 项目结构

```
llm-from-scratch/
├── README.md                       # 语言选择入口 / language picker
├── docs/                           # 多语言完整文档 / full docs per language
│   ├── readme-en.md
│   ├── readme-zh.md
│   └── readme-id.md
├── requirements.txt                # Python 依赖
├── configs/                        # YAML 预设（参考用）
│   ├── tiny.yaml
│   ├── 0.1b.yaml
│   ├── 0.5b.yaml
│   ├── 1b.yaml
│   └── custom.yaml
├── llm_scratch/                    # Python 包
│   ├── __init__.py
│   ├── config.py                   # 尺寸预设与配置构建
│   ├── model_gpt2.py               # GPT-2 风格模型
│   ├── model_llama.py              # LLaMA 风格模型
│   ├── dataset.py                  # WikiText-103 + tokenizer
│   ├── train.py                    # 训练入口
│   └── generate.py                 # 生成脚本
└── notebooks/
    └── train_llm_colab.ipynb       # 一键 Colab 笔记本
```

---

## 🧮 尺寸预设

| `size` | n_layer | n_head | n_embd | block_size | ~参数量 | 默认 batch×accum | 推荐 Colab |
|--------|---------|--------|--------|------------|---------|----------------------|--------------------|
| `tiny`    | 4  | 4  | 256  | 256  | ~16M  | 32 × 1  | 免费 T4 ✅ |
| `0.1b`    | 8  | 12 | 768  | 512  | ~95M  | 12 × 2  | 免费 T4（慢）/ Pro ✅ |
| `0.5b`    | 24 | 16 | 1280 | 1024 | ~500M | 4 × 8   | Pro / A100 ✅ |
| `1b`      | 24 | 16 | 1536 | 1024 | ~750M | 2 × 16  | 仅 A100 ⚠️ |
| `custom`  | 自定 |    |      |      |       | 16 × 2  | 视情况而定 |

参数量按 `vocab_size=50257` 估算；LLaMA 风格因 SwiGLU 圆整每层略小。`0.5b`
及以上请开启 `grad_checkpoint=True` 以适配有限显存。

---

## 🚀 Colab 快速开始

1. 打开 [Google Colab](https://colab.research.google.com/)。
2. **运行时 → 更改运行时类型 → T4 GPU**（Pro 用户可选 A100）。
3. 上传本项目目录到 Colab：建议先打包成 zip 上传，再 `!unzip` 解压。
4. 在 Colab 中打开 `notebooks/train_llm_colab.ipynb`（**文件 → 上传笔记本**）。
5. 从上到下依次运行单元格。**在第 4 节设置 `ARCH` 与 `SIZE`** —— 这就是"选择参数"。

### 冒烟测试（免费 T4，约 10 分钟）

```python
ARCH = "gpt2"
SIZE = "tiny"
TRAIN = {"epochs": 1, "subset_size": 2000}
```

### 正式训练

```python
ARCH = "llama"
SIZE = "0.1b"
TRAIN = {"epochs": 1}  # 完整 WikiText-103
```

---

## 💻 本地使用

### 安装

```bash
git clone https://github.com/BF667/llm-from-scratch.git
cd llm-from-scratch
pip install -r requirements.txt
```

### 训练

```bash
# Tiny GPT-2 + 2000 行子集冒烟测试
python -m llm_scratch.train --arch gpt2 --size tiny --output_dir out/gpt2-tiny \
    --subset_size 2000 --epochs 1

# LLaMA 0.1B 完整 WikiText-103
python -m llm_scratch.train --arch llama --size 0.1b --output_dir out/llama-0.1b

# 自定义尺寸 —— 覆盖单个参数
python -m llm_scratch.train --arch gpt2 --size custom \
    --n_layer 12 --n_head 12 --n_embd 768 --block_size 768 \
    --output_dir out/gpt2-custom --grad_checkpoint

# 关闭混合精度
python -m llm_scratch.train --arch gpt2 --size tiny --no_fp16
```

### 生成

```bash
python -m llm_scratch.generate --checkpoint out/gpt2-tiny \
    --prompt "The future of AI is" --max_new_tokens 80
```

### 作为 Python 库使用

```python
from llm_scratch import build_model, build_config, list_presets

# 显示所有预设
print(list_presets())

# 构建 0.5B LLaMA
model = build_model(arch="llama", size="0.5b")
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# 自定义尺寸
model = build_model(arch="gpt2", size="custom",
                    n_layer=12, n_head=12, n_embd=768, block_size=768)
```

---

## 🧠 架构细节

### GPT-2 风格（`model_gpt2.py`）

| 组件 | 选择 |
|-----------|--------|
| Norm | LayerNorm（pre-norm） |
| 位置编码 | 学习式绝对位置 embedding（`nn.Embedding(block_size, n_embd)`） |
| Attention | 多头因果注意力，融合 QKV，可用时启用 Flash Attention |
| MLP | 4× 扩展，GELU |
| LM head | 与 token embedding 权重绑定 |
| Bias | 默认关闭（现代风格） |
| Dropout | 可配置（默认 0.0） |

### LLaMA 风格（`model_llama.py`）

| 组件 | 选择 |
|-----------|--------|
| Norm | RMSNorm（不减均值，无 bias） |
| 位置编码 | 旋转位置编码（RoPE） |
| Attention | 多头因果 + RoPE 施加于 Q/K，可用时启用 Flash Attention |
| MLP | SwiGLU（`w2(silu(w1(x)) * w3(x))`），hidden 维度按 `multiple_of=256` 圆整 |
| LM head | 与 embedding 解绑（`tie_word_embeddings=False`） |
| Bias | 全部关闭 |
| Dropout | 默认 0.0 |

两个模型均注册为 `PreTrainedModel` 子类，原生支持 `.save_pretrained()` /
`.from_pretrained()` / 梯度检查点 / 混合精度 / HF `Trainer`。

---

## ⚙️ 训练默认值

| 参数 | 默认值 | 说明 |
|------|---------|-------|
| 优化器 | `adamw_torch` | AdamW，解耦权重衰减 |
| 学习率 | `3e-4` | 各尺寸相同；cosine 衰减至 0 |
| Warmup | 总步数的 5% | 线性 warmup |
| 权重衰减 | `0.1` | 应用于所有非 norm / 非 bias 参数 |
| 学习率调度 | Cosine | 训练结束衰减至 0 |
| 混合精度 | 支持 bf16 则用 bf16，否则 fp16 | 无 CUDA 时禁用 |
| 梯度检查点 | 关闭 | 用 `--grad_checkpoint` 或 `model.gradient_checkpointing_enable()` 开启 |
| 评估 | 每 500 步 | 在 `validation` 划分上 |
| 保存 | 每 500 步，仅保留最近 2 个 | |

---

## 📊 预期结果

在 **WikiText-103** 上训练 **1 epoch**：

| size | arch | train loss | val loss | ppl | T4 耗时 | A100 耗时 |
|------|------|-----------|----------|-----|------------|--------------|
| tiny | gpt2 | ~4.5 | ~4.8 | ~120 | ~10 分钟 | ~3 分钟 |
| 0.1b | gpt2 | ~3.4 | ~3.7 | ~40 | ~1.5 小时 | ~20 分钟 |
| 0.1b | llama | ~3.3 | ~3.6 | ~36 | ~2 小时 | ~25 分钟 |
| 0.5b | llama | ~2.9 | ~3.2 | ~24 | T4 OOM | ~2 小时 |

上述为粗略估计，实际数值随 `block_size`、`batch_size`、`grad_accum` 与 Colab
负载波动。`tiny` 难以产出连贯英文，至少需 `0.1b` 才能生成可读文本。

---

## 🧪 提示与注意事项

- **首次运行会下载 WikiText-103**（约 180MB）。后续运行使用缓存。
- **Colab 免费 T4 显存 16GB**。`0.5b` 必须开启梯度检查点 + 小 batch + 梯度累积。`1b` 在 Colab 上仅作演示 —— 建议使用 A100 或租用云 GPU。
- **若要真正训练可用 LLM**，本项目为教学脚手架。需要：更多数据（如 FineWeb-Edu）、更长上下文、更多训练步数、更大词表。本项目给你骨架，按需替换。
- **可复现性**：设置 `--seed`（暂未暴露 —— 如需要可加到 `TrainingArguments`）。
- **多 GPU**：当前 `Trainer` 配置支持单 GPU。如需 DDP，用 `accelerate launch` 包装。

---

## 📜 许可证

MIT —— 随便用，模型胡说别找我。
