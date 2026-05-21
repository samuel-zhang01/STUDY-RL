"""Generate the figures embedded in notes.pdf.

Deterministic given a seed.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.plotting import use_rl_style, RL_PALETTE
from common.seed import seed_everything

from .train import train


def main(seed: int = 0, out_dir: str | None = None) -> None:
    use_rl_style()
    seed_everything(seed)
    here = os.path.dirname(__file__)
    out_dir = out_dir or os.path.join(here, "..", "figs")
    os.makedirs(out_dir, exist_ok=True)

    curves = {algo: train(seed=seed, algo=algo, n_steps=2000) for algo in ("naive", "cql", "iql")}

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colours = {"naive": RL_PALETTE["red"], "cql": RL_PALETTE["orange"], "iql": RL_PALETTE["teal"]}
    for algo, out in curves.items():
        ev = out["eval_returns"]
        ax.plot(ev[:, 0], ev[:, 1], label=f"{algo} (final={out['final_return']:.2f})",
                color=colours[algo])
    ax.axhline(2.18, ls="--", color=RL_PALETTE["gray"], label="optimal eval")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("greedy-policy return (eval)")
    ax.set_title("Offline RL on synthetic GridWorld: extrapolation error vs conservatism")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_offline_curves.pdf"))
    plt.close(fig)

    # Heatmap of learned Q values for IQL vs naive.
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    for ax, algo in zip(axes, ("naive", "cql", "iql")):
        Q = curves[algo]["Q"].max(axis=1).reshape(5, 5)
        im = ax.imshow(Q, cmap="viridis")
        ax.set_title(f"max_a Q(s,a)  [{algo}]")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_q_heatmaps.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
