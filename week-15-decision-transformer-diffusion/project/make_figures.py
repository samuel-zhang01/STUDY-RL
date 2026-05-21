"""Figures for Week 15 notes.pdf."""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.plotting import use_rl_style, RL_PALETTE
from common.seed import seed_everything

from .train import train
from .solutions import exercise_6_action_distribution


def main(seed: int = 0, out_dir: str | None = None) -> None:
    use_rl_style(); seed_everything(seed)
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "..", "figs")
    os.makedirs(out_dir, exist_ok=True)

    out = train(seed=seed, n_steps=1200)
    losses = out["losses"]

    # Loss curve.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    win = 25
    smooth = np.convolve(losses, np.ones(win) / win, mode="valid")
    ax.plot(np.arange(len(smooth)), smooth, color=RL_PALETTE["teal"])
    ax.set_xlabel("gradient step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Decision Transformer: action-prediction loss (smoothed)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_dt_loss.pdf")); plt.close(fig)

    # Conditional-generation contrast.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(["target=-5", "target=+3"],
           [out["mean_return_low_target"], out["mean_return_high_target"]],
           color=[RL_PALETTE["red"], RL_PALETTE["teal"]])
    ax.set_ylabel("mean evaluation return")
    ax.set_title("Return-conditioned generation: DT respects the target")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_dt_conditional.pdf")); plt.close(fig)

    # Action distribution at s=0.
    p = exercise_6_action_distribution(out)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(["up", "right", "down", "left"], p, color=RL_PALETTE["blue"])
    ax.set_ylabel("p(a | s=0, R=+3)")
    ax.set_title("DT next-action distribution at the start state")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_dt_action_dist.pdf")); plt.close(fig)


if __name__ == "__main__":
    main()
