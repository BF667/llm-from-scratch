"""
Generate text from a trained checkpoint.
从训练好的 checkpoint 生成文本。

Usage:
    python -m llm_scratch.generate --checkpoint out/gpt2-tiny --prompt "The future of AI is" --max_new_tokens 80
"""

from __future__ import annotations
import argparse
import torch

from transformers import AutoTokenizer

from .model_gpt2 import GPTScratchForCausalLM, GPTScratchConfig
from .model_llama import LlamaScratchForCausalLM, LlamaScratchConfig


def load_model(checkpoint_dir: str):
    cfg_gpt2 = GPTScratchConfig.from_pretrained(checkpoint_dir)
    if cfg_gpt2.model_type == "gpt_scratch":
        model = GPTScratchForCausalLM.from_pretrained(checkpoint_dir)
    else:
        cfg_llama = LlamaScratchConfig.from_pretrained(checkpoint_dir)
        model = LlamaScratchForCausalLM.from_pretrained(checkpoint_dir)
    return model


def main():
    p = argparse.ArgumentParser(
        description="Generate text from a trained from-scratch LLM. "
                    "从训练好的模型生成文本。"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained model dir / 模型目录")
    p.add_argument("--prompt", default="The future of AI is",
                   help="Prompt text / 提示文本")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--no_sample", action="store_true",
                   help="Greedy decoding / 贪心解码")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = load_model(args.checkpoint).to(device)
    model.eval()

    ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"\n[Prompt] {args.prompt}\n[Output]")
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.no_sample,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
