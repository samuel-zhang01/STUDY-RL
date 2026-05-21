"""Figures for Week 17."""
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


def main(seed: int = 0, out_dir: str | None = None) -> None:
    use_rl_style(); seed_everything(seed)
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "..", "figs")
    os.makedirs(out_dir, exist_ok=True)
    out = train(seed=seed)

    # Visit heatmaps.
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, name in zip(axes, ("rnd_off", "rnd_on")):
        v = out[name]["visits"]
        im = ax.imshow(np.log1p(v), cmap="magma")
        ax.set_title(name.replace("_", " ") + f"  goals={out[name]['goal_hits']}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_rnd_visits.pdf"))
    plt.close(fig)

    # Goal-rate curve.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, color in (("rnd_off", RL_PALETTE["red"]), ("rnd_on", RL_PALETTE["teal"])):
        r = out[name]["returns"]
        # running success rate
        win = 20
        rate = np.convolve(r > 0, np.ones(win) / win, mode="valid")
        ax.plot(rate, label=name, color=color)
    ax.set_xlabel("episode (smoothed)"); ax.set_ylabel("success rate")
    ax.set_title("RND vs no-RND on a sparse-reward 7x7 grid")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_rnd_curve.pdf"))
    plt.close(fig)

    # MAML pre/post adaptation curve.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    gap = out["maml"]["pre_post_gap"]
    ax.plot(gap[:, 0], label="pre-adapt mean reward", color=RL_PALETTE["red"])
    ax.plot(gap[:, 1], label="post-adapt mean reward", color=RL_PALETTE["teal"])
    ax.set_xlabel("outer step"); ax.set_ylabel("expected reward")
    ax.set_title("MAML meta-bandit: adaptation gap grows over training")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_maml.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
