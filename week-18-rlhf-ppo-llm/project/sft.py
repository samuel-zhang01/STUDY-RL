"""Supervised fine-tuning — stage 1 of RLHF."""
from __future__ import annotations
from typing import List, Sequence
import torch
import torch.nn.functional as F

from .model import CharTransformer
from .data import encode, STOI, VOCAB_SIZE


def _pad(seqs: Sequence[List[int]], block: int) -> torch.Tensor:
    out = torch.zeros(len(seqs), block, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:block]
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def sft_train(
    model: CharTransformer,
    strings: Sequence[str],
    steps: int = 200,
    lr: float = 3e-3,
    batch_size: int = 16,
    device: str = "cpu",
    seed: int = 0,
):
    """Tiny SFT loop: causal-LM cross-entropy on (prompt + chosen).

    Returns a list of per-step losses for plotting.
    """
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    block = model.cfg.block_size
    enc = [encode(s) for s in strings]
    losses = []
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, len(enc), (batch_size,), generator=g)
        batch = _pad([enc[int(i)] for i in idx], block).to(device)
        logits = model(batch[:, :-1])  # (B, T-1, V)
        targets = batch[:, 1:]
        # Ignore pad tokens (idx 0) in the loss.
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=0,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    return losses
