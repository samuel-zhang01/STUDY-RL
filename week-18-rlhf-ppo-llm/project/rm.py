"""Reward model — stage 2 of RLHF.

We implement the Bradley--Terry pairwise loss

    L = - log sigmoid( r(x, y_w) - r(x, y_l) )

over a dataset of preference triples (prompt, chosen, rejected).
"""
from __future__ import annotations
from typing import Sequence, Tuple
import torch
import torch.nn.functional as F

from .model import RewardModel
from .data import encode


def _pad_pair(seqs, block):
    out = torch.zeros(len(seqs), block, dtype=torch.long)
    lens = torch.zeros(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:block]
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        lens[i] = len(s)
    return out, lens


def reward_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """Bradley--Terry cross-entropy: -log sigmoid(r_w - r_l)."""
    return -F.logsigmoid(r_w - r_l).mean()


def rm_train(
    rm: RewardModel,
    pairs: Sequence[Tuple[str, str, str]],
    steps: int = 200,
    lr: float = 3e-3,
    batch_size: int = 16,
    device: str = "cpu",
    seed: int = 0,
):
    """Train the reward model on (prompt, chosen, rejected) triples."""
    torch.manual_seed(seed)
    rm.to(device).train()
    opt = torch.optim.AdamW(rm.parameters(), lr=lr)
    block = rm.backbone.cfg.block_size

    enc_w = [encode(p + c) for (p, c, _) in pairs]
    enc_l = [encode(p + l) for (p, _, l) in pairs]
    losses, accs = [], []
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, len(pairs), (batch_size,), generator=g)
        w_ids, w_len = _pad_pair([enc_w[int(i)] for i in idx], block)
        l_ids, l_len = _pad_pair([enc_l[int(i)] for i in idx], block)
        w_ids, l_ids = w_ids.to(device), l_ids.to(device)
        w_len, l_len = w_len.to(device), l_len.to(device)
        r_w = rm(w_ids, w_len)
        r_l = rm(l_ids, l_len)
        loss = reward_loss(r_w, r_l)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        accs.append(float((r_w > r_l).float().mean().item()))
    return losses, accs
