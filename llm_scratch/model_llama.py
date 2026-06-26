"""
LLaMA-style decoder-only LLM from scratch on HF Transformers.
基于 HuggingFace Transformers 从头实现的 LLaMA 风格 decoder-only LLM。

Modern upgrades vs GPT-2:
  - RMSNorm instead of LayerNorm
  - Rotary positional embeddings (RoPE) instead of learned absolute positions
  - SwiGLU MLP instead of GELU MLP
  - No bias anywhere
  - No dropout (LLaMA convention; add via config if you want)
"""

from __future__ import annotations
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class LlamaScratchConfig(PretrainedConfig):
    model_type = "llama_scratch"

    def __init__(
        self,
        vocab_size: int = 50257,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        block_size: int = 256,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        tie_word_embeddings: bool = False,    # LLaMA does NOT tie embeddings
        multiple_of: int = 256,               # for SwiGLU hidden dim rounding
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.multiple_of = multiple_of
        self.rms_norm_eps = rms_norm_eps


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm — LLaMA's choice. No mean subtraction, no bias.
    LLaMA 使用的 RMSNorm，不减均值，无 bias。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex rotation frequencies for RoPE up to length `end`.
    预计算 RoPE 的复数旋转频率。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 (end, dim/2)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply RoPE to query and key tensors.
    对 query/key 张量施加 RoPE。

    xq, xk: (B, n_head, T, head_dim)
    freqs_cis: (T, head_dim/2) complex64
    """
    B, H, T, D = xq.shape
    # view as complex: (B, H, T, D/2)
    xq_ = torch.view_as_complex(xq.float().reshape(B, H, T, D // 2, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(B, H, T, D // 2, 2))
    freqs_cis = freqs_cis.view(1, 1, T, D // 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(B, H, T, D)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(B, H, T, D)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""
    def __init__(self, cfg: LlamaScratchConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.training and self.dropout > 0:
                att = F.dropout(att, p=self.dropout)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class SwiGLU(nn.Module):
    """SwiGLU MLP — LLaMA's choice. hidden dim rounded to multiple_of.
    LLaMA 使用的 SwiGLU MLP，hidden 维度按 multiple_of 圆整。"""
    def __init__(self, cfg: LlamaScratchConfig):
        super().__init__()
        hidden = int(2 * (4 * cfg.n_embd) / 3)
        hidden = ((hidden + cfg.multiple_of - 1) // cfg.multiple_of) * cfg.multiple_of
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)   # gate
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)   # down
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)   # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """LLaMA pre-norm block."""
    def __init__(self, cfg: LlamaScratchConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class LlamaScratchForCausalLM(PreTrainedModel):
    """LLaMA-style causal LM, ready for HF Trainer.
    LLaMA 风格 causal LM，可直接用 HF Trainer 训练。"""

    config_class = LlamaScratchConfig
    base_model_prefix = "llama_scratch"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlamaScratchConfig):
        super().__init__(config)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        if config.tie_word_embeddings:
            self.lm_head = None  # use tok_emb.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # RoPE precompute (buffer, not parameter)
        freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size,
            config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)
        # Scale residual init per GPT-2/LLaMA convention.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        self.post_init()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaScratchForCausalLM):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        # slice RoPE to current sequence length
        freqs_cis = self.freqs_cis[:T]

        gc = getattr(self, "gradient_checkpointing", False) and self.training
        for block in self.blocks:
            if gc:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, freqs_cis, use_reentrant=False
                )
            else:
                x = block(x, freqs_cis)

        x = self.norm(x)
        if self.lm_head is None:
            logits = x @ self.tok_emb.weight.t()
        else:
            logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        self.eval()
        block_size = self.config.block_size
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= block_size else input_ids[:, -block_size:]
            logits = self.forward(input_ids=idx_cond).logits[:, -1, :]
            if do_sample and temperature > 0:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
