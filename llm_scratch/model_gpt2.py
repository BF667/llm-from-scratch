"""
GPT-2 style decoder-only LLM, implemented from scratch on top of HF Transformers.
基于 HuggingFace Transformers 从头实现的 GPT-2 风格 decoder-only LLM。

Why register with HF AutoClasses? So we can use `Trainer`, `save_pretrained`,
`from_pretrained`, generation utils, etc. just like a stock HF model.
"""

from __future__ import annotations
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class GPTScratchConfig(PretrainedConfig):
    """Config for our from-scratch GPT-2 style model.
    从头实现的 GPT-2 风格模型配置。"""

    model_type = "gpt_scratch"

    def __init__(
        self,
        vocab_size: int = 50257,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        block_size: int = 256,
        dropout: float = 0.0,
        bias: bool = False,           # following LLaMA/GPT-2 modern style: no bias
        activation: str = "gelu_new",
        tie_word_embeddings: bool = True,
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
        self.dropout = dropout
        self.bias = bias
        self.activation = activation


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention.
    标准多头因果自注意力。"""

    def __init__(self, cfg: GPTScratchConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        # fused QKV
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # flash attention when available
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.training and self.dropout > 0:
                att = F.dropout(att, p=self.dropout)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTScratchConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.act = nn.GELU() if cfg.activation == "gelu" else nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(self.fc(x)))


class Block(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, cfg: GPTScratchConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class GPTScratchForCausalLM(PreTrainedModel):
    """GPT-2 style causal LM with tied embeddings, ready for HF Trainer.
    GPT-2 风格 causal LM，权重绑定，可直接用 HF Trainer 训练。"""

    config_class = GPTScratchConfig
    base_model_prefix = "gpt_scratch"
    supports_gradient_checkpointing = True

    def __init__(self, config: GPTScratchConfig):
        super().__init__(config)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        # LM head is tied with tok_emb when tie_word_embeddings=True
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        else:
            self.lm_head = None  # use tok_emb.weight
        # init
        self.apply(self._init_weights)
        self.post_init()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- gradient checkpointing plumbing ------------------------------------
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTScratchForCausalLM):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, T = input_ids.shape
        device = input_ids.device

        # position ids
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # gradient checkpointing optionally
        gc = getattr(self, "gradient_checkpointing", False) and self.training
        for block in self.blocks:
            if gc:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = (x @ self.tok_emb.weight.t()) if self.lm_head is None else self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift: predict next token. labels already shifted externally? No — do it here.
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

    # -- generation helper --------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """Simple top-k sampling generate(). Not as fancy as HF .generate() but
        self-contained and works without generation_config plumbing.
        简单 top-k 采样生成，无需 generation_config 即可使用。"""
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
