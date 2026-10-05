"""Generate all figures used in notes.pdf.

Deterministic given a seed. Output PDFs land in `figs/`.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
WEEK = os.path.dirname(HERE)
ROOT = os.path.dirname(WEEK)
sys.path.insert(0, ROOT)
sys.path.insert(0, WEEK)

import numpy as np
import matplotlib.pyplot as plt

from common.plotting import use_rl_style, RL_PALETTE
from common.seed import seed_everything

from project.constrained import train_lagrangian
from project.env import generate_population, true_match_prob
from project.scorer import heuristic_score
from project.train import train

FIGS = os.path.join(WEEK, "figs")
os.makedirs(FIGS, exist_ok=True)


def fig_ope_estimates(seed: int = 0) -> str:
    use_rl_style()
    res = train(n_candidates=400, n_jobs=60, n_rounds=1500, seed=seed)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    methods = ["Heuristic", "Uniform", "LinTS"]
    snips = [res.snips_estimates[m] for m in methods]
    dr = [res.dr_estimates[m] for m in methods]
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w / 2, snips, w, label="SNIPS", color=RL_PALETTE["teal"])
    ax.bar(x + w / 2, dr, w, label="DR", color=RL_PALETTE["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Estimated hire rate")
    ax.set_title("Fresh-log OPE: heuristic vs posterior-mean reranker")
    ax.legend()
    out = os.path.join(FIGS, "fig_ope.pdf")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_parity(seed: int = 0) -> str:
    use_rl_style()
    data = generate_population(n_candidates=300, n_jobs=40, seed=seed)
    stats = train_lagrangian(data, n_rounds=500, k_slate=10, seed=seed)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(stats.parity_gap, color=RL_PALETTE["blue"], label="B/A selected-group share")
    ax.axhline(0.8, ls="--", color=RL_PALETTE["red"], label="toy share-ratio floor")
    ax2 = ax.twinx()
    ax2.plot(stats.lam_hist, color=RL_PALETTE["orange"], alpha=0.7, label="Lagrange $\\lambda$")
    ax.set_xlabel("Round")
    ax.set_ylabel("Parity ratio (B/A)")
    ax2.set_ylabel("$\\lambda$")
    ax.set_title("Heuristic score penalty with a dual update")
    ax.legend(loc="lower right")
    out = os.path.join(FIGS, "fig_parity.pdf")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_score_distribution(seed: int = 0) -> str:
    use_rl_style()
    data = generate_population(n_candidates=400, n_jobs=60, seed=seed)
    s = heuristic_score(data)
    p = 100.0 * true_match_prob(data)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.scatter(s.ravel()[:4000], p.ravel()[:4000], s=4, alpha=0.4, color=RL_PALETTE["teal"])
    ax.set_xlabel("Heuristic score (0--100)")
    ax.set_ylabel("True match probability $\\times$ 100")
    ax.set_title("Heuristic vs ground truth (each dot is one (cand, job) pair)")
    out = os.path.join(FIGS, "fig_score_calibration.pdf")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(seed: int = 0):
    seed_everything(seed)
    paths = [fig_ope_estimates(seed), fig_parity(seed), fig_score_distribution(seed)]
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
