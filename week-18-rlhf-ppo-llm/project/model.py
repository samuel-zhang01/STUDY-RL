"""A tiny char-level transformer for week 18.

Small enough to train in seconds on CPU; structured so that a single
backbone is shared between the policy (with an LM head) and the reward
model (with a scalar head).  This mirrors how production RLHF code
shares the trunk between actor and critic to save memory.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import VOCAB_SIZE


@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    """Decoder-only transformer with an optional value head."""

    def __init__(self, cfg: TransformerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        self.tok = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos = nn.Embedding(self.cfg.block_size, self.cfg.d_model)
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.n_layer))
        self.ln_f = nn.LayerNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        # Tie weights — saves params and helps tiny models.
        self.lm_head.weight = self.tok.weight

    def trunk(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq {T} > block {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        h = self.tok(idx) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.ln_f(h)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.trunk(idx))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class ValueHead(nn.Module):
    """A scalar head bolted onto the transformer trunk."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).squeeze(-1)


class RewardModel(nn.Module):
    """A backbone + scalar head whose output at the last non-pad token
    is the scalar reward for the whole completion."""

    def __init__(self, backbone: CharTransformer) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = ValueHead(backbone.cfg.d_model)

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        h = self.backbone.trunk(idx)  # (B, T, d)
        scores = self.head(h)  # (B, T)
        if lengths is None:
            # Use last position.
            return scores[:, -1]
        # Gather the score at the last real token for each row.
        idx_last = (lengths - 1).clamp(min=0).long()
        return scores[torch.arange(scores.size(0), device=scores.device), idx_last]
