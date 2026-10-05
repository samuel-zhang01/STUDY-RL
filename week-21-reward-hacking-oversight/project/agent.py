"""Tiny char-level LM shared between the hack-lab and CAI pipelines.

This is the same architecture as week 20's ``CharLM`` but trimmed and
copied here so each week is self-contained.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_CHARS = list("abcdefghijklmnopqrstuvwxyz .,!") + ["<bos>", "<eos>", "<pad>"]


def build_tokenizer() -> tuple[dict[str, int], dict[int, str]]:
    stoi = {c: i for i, c in enumerate(VOCAB_CHARS)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[c] for c in text if c in stoi]


def decode(ids: Iterable[int], itos: dict[int, str]) -> str:
    out = []
    for i in ids:
        tok = itos[int(i)]
        if tok in ("<bos>", "<pad>"):
            continue
        if tok == "<eos>":
            break
        out.append(tok)
    return "".join(out)


@dataclass
class CharLMConfig:
    vocab_size: int
    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 64
    dropout: float = 0.0


class _Attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).view(
                1, 1, cfg.max_seq_len, cfg.max_seq_len),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class _Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = _Attn(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CharLM(nn.Module):
    def __init__(self, cfg: CharLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens, temperature=1.0, eos_id=None):
        ids = prefix.clone()
        B = ids.shape[0]
        logps = torch.zeros(B, max_new_tokens, device=ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        for t in range(max_new_tokens):
            inp = ids[:, -self.cfg.max_seq_len:]
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
                pad_len = max_new_tokens - t - 1
                if pad_len > 0:
                    pad = torch.full((B, pad_len), eos_id if eos_id is not None else 0,
                                     dtype=ids.dtype, device=ids.device)
                    ids = torch.cat([ids, pad], dim=-1)
                break
        return ids, logps


def completion_mask(comp_ids, eos_id):
    is_eos = comp_ids == eos_id
    after_eos = is_eos.cumsum(dim=-1) - is_eos.long()
    return (after_eos == 0).float()


def token_logprobs(model, ids, comp_start, temperature=1.0):
    logits = model(ids[:, :-1]) / temperature
    lp = F.log_softmax(logits, dim=-1)
    targets = ids[:, 1:]
    chosen = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return chosen[:, comp_start - 1:]
