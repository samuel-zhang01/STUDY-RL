"""Tiny character-level Transformer policy for week 20.

This is *not* a production LM. It is the smallest causal Transformer that can
fit GRPO's mechanics: token-level log-probs, KL-to-reference, and an autoregressive
sampler. By keeping the vocabulary at ~20 characters and the model under
50k parameters, the entire training loop runs CPU-friendly.

The design is intentionally close to the HF interface, so the patterns
generalise: ``logits = model(input_ids)``; sampling uses temperature + categorical;
KL is computed token-wise between current and reference log-probs.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Tokenizer — a fixed character vocab. We expose <bos>, <eos>, <pad>.
# ----------------------------------------------------------------------------

VOCAB_CHARS = list("0123456789+-=. ") + ["<bos>", "<eos>", "<pad>"]


def build_tokenizer() -> tuple[dict[str, int], dict[int, str]]:
    stoi = {c: i for i, c in enumerate(VOCAB_CHARS)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[c] for c in text]


def decode(ids: Iterable[int], itos: dict[int, str]) -> str:
    out: list[str] = []
    for i in ids:
        tok = itos[int(i)]
        if tok in ("<bos>", "<pad>"):
            continue
        if tok == "<eos>":
            break
        out.append(tok)
    return "".join(out)


# ----------------------------------------------------------------------------
# Tiny causal Transformer.
# ----------------------------------------------------------------------------


@dataclass
class CharLMConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 32
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: CharLMConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, cfg.max_seq_len, cfg.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: CharLMConfig) -> None:
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


class CharLM(nn.Module):
    """Tiny GPT-style decoder. ``forward`` returns logits over the vocabulary.

    The model exposes :meth:`generate` which performs temperature sampling
    and returns ``(seq, log_probs_per_token)`` so the RL loop has the on-policy
    log-probabilities at *sampling time* — exactly the quantity GRPO and PPO
    need to compute the importance ratio.
    """

    def __init__(self, cfg: CharLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight-tying — saves params and stabilises tiny LMs.
        self.head.weight = self.tok_emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a continuation. Returns ``(full_ids, sample_logprobs)``.

        ``sample_logprobs`` has shape ``(B, max_new_tokens)`` with -inf
        positions masked to 0 once EOS has been emitted — those positions are
        also masked out in the loss via :func:`completion_mask`.
        """
        ids = prefix.clone()
        B = ids.shape[0]
        logps = torch.zeros(B, max_new_tokens, device=ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        for t in range(max_new_tokens):
            inp = ids[:, -self.cfg.max_seq_len :]
            logits = self.forward(inp)[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            lp = torch.log(probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1) + 1e-12)
            lp = torch.where(done, torch.zeros_like(lp), lp)
            logps[:, t] = lp
            ids = torch.cat([ids, sampled.unsqueeze(-1)], dim=-1)
            if eos_id is not None:
                done = done | (sampled == eos_id)
            if done.all():
                # pad the rest with EOS to keep shapes uniform.
                pad_len = max_new_tokens - t - 1
                if pad_len > 0:
                    pad = torch.full(
                        (B, pad_len), eos_id if eos_id is not None else 0,
                        dtype=ids.dtype, device=ids.device,
                    )
                    ids = torch.cat([ids, pad], dim=-1)
                break
        return ids, logps


def completion_mask(comp_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """1 up to and including the first EOS in each row, 0 thereafter."""
    is_eos = comp_ids == eos_id
    # cumsum after first EOS becomes >=1 strictly after that position
    after_eos = is_eos.cumsum(dim=-1) - is_eos.long()
    return (after_eos == 0).float()


def token_logprobs(model: CharLM, ids: torch.Tensor, comp_start: int) -> torch.Tensor:
    """Return per-token log-probs for the *completion* slice of ``ids``.

    For ``ids`` of shape ``(B, T)`` with the first ``comp_start`` tokens being
    the prompt, returns log-probs of shape ``(B, T - comp_start)`` corresponding
    to positions ``[comp_start, T)`` (i.e., predicting the i-th completion
    token given everything before it).
    """
    logits = model(ids[:, :-1])  # predict position t+1 from prefix [0..t]
    log_probs_all = F.log_softmax(logits, dim=-1)
    targets = ids[:, 1:]  # (B, T-1)
    chosen = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # the completion targets are positions [comp_start, T) in ids,
    # which are slots [comp_start - 1, T - 1) in the chosen tensor.
    return chosen[:, comp_start - 1 :]
