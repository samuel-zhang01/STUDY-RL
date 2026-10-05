"""A small two-tower retrieval model: candidate-tower encodes CV (skills bag);
job-tower encodes JD (required + nice-to-have bags). Both project into a
shared d-dim embedding; relevance is the cosine similarity.

Trained with in-batch-negatives contrastive loss (the dominant 2024-2026
recipe for dense retrieval). CPU-only, <30s training, dataset of 200 cands
times 50 jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    """Two small MLPs that share an output dimension."""

    def __init__(self, cand_dim: int, job_dim: int, embed_dim: int = 16,
                 hidden: int = 32) -> None:
        super().__init__()
        self.cand_tower = nn.Sequential(
            nn.Linear(cand_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.job_tower = nn.Sequential(
            nn.Linear(job_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def encode_candidate(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cand_tower(x)
        return F.normalize(z, dim=-1)

    def encode_job(self, y: torch.Tensor) -> torch.Tensor:
        z = self.job_tower(y)
        return F.normalize(z, dim=-1)

    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (self.encode_candidate(x) * self.encode_job(y)).sum(-1)


def train_two_tower(
    ds,
    embed_dim: int = 16,
    hidden: int = 32,
    epochs: int = 80,
    batch_size: int = 32,
    temperature: float = 0.1,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[TwoTower, np.ndarray]:
    """Train symmetric InfoNCE using each candidate's best synthetic job.

    Targets are diagonal class indices, not soft relevance distributions.
    Duplicate jobs within a batch are treated as negatives in this simplified
    recipe; a production retriever needs explicit multiple-positive handling.

    Returns (model, loss_history).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    cand_dim = ds.cand_vecs.shape[1]
    job_dim = ds.job_req_vecs.shape[1] + ds.job_nice_vecs.shape[1]
    model = TwoTower(cand_dim, job_dim, embed_dim=embed_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Build (candidate, best-job) positive pairs from ground truth.
    M = ds.true_match_matrix()  # (n_jobs, n_candidates)
    best_job_per_cand = M.argmax(axis=0)  # for each cand, best matching job
    cand_X = torch.tensor(ds.cand_vecs, dtype=torch.float32, device=device)
    job_X = torch.tensor(
        np.concatenate([ds.job_req_vecs, ds.job_nice_vecs], axis=1),
        dtype=torch.float32, device=device,
    )

    losses = []
    n_cands = ds.cand_vecs.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n_cands)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_cands, batch_size):
            idx = order[start:start + batch_size]
            jids = best_job_per_cand[idx]
            x = cand_X[idx]
            y = job_X[jids]
            # Symmetric InfoNCE.
            z_c = model.encode_candidate(x)        # (B, d)
            z_j = model.encode_job(y)              # (B, d)
            logits = z_c @ z_j.t() / temperature   # (B, B)
            targets = torch.arange(x.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, targets) +
                          F.cross_entropy(logits.t(), targets))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        losses.append(ep_loss / max(n_batches, 1))
    return model, np.array(losses)


def topk_for_job(model: TwoTower, ds, jid: int, k: int = 10) -> np.ndarray:
    """Return the top-k candidate ids for job `jid` under the two-tower model."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(ds.cand_vecs, dtype=torch.float32)
        y = torch.tensor(
            np.concatenate([ds.job_req_vecs[jid], ds.job_nice_vecs[jid]]),
            dtype=torch.float32,
        ).unsqueeze(0)
        z_c = model.encode_candidate(x)
        z_j = model.encode_job(y)
        scores = (z_c @ z_j.t()).squeeze(-1).cpu().numpy()
    order = np.argsort(-scores)
    return order[:k]


def all_scores(model: TwoTower, ds) -> np.ndarray:
    """Return the (n_jobs, n_candidates) two-tower score matrix."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(ds.cand_vecs, dtype=torch.float32)
        y = torch.tensor(
            np.concatenate([ds.job_req_vecs, ds.job_nice_vecs], axis=1),
            dtype=torch.float32,
        )
        z_c = model.encode_candidate(x)
        z_j = model.encode_job(y)
        scores = (z_j @ z_c.t()).cpu().numpy()
    return scores
