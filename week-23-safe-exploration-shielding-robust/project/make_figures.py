"""Generate week-23 figures.

Produces (under figs/):
    fig_grid.pdf              — DangerGridWorld layout
    fig_dfa.pdf               — the two-state safety automaton
    fig_unsafe_curve.pdf      — cumulative unsafe transitions, shielded vs not
    fig_return_curve.pdf      — episodic return, shielded vs not
    fig_pendulum_dr.pdf       — eval-return-vs-length, DR vs fixed
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402

from .env_grid import DangerGridWorld  # noqa: E402
from .q_learn import train_shielded, train_unshielded  # noqa: E402
from .pendulum_train import train_pendulum_dr, PendulumTrainConfig  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "figs"


def fig_grid():
    use_rl_style()
    env = DangerGridWorld()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, env.cols); ax.set_ylim(0, env.rows); ax.set_aspect("equal")
    ax.invert_yaxis()
    for (r, c) in env.danger:
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=RL_PALETTE["red"], alpha=0.7))
        ax.text(c + 0.5, r + 0.5, "D", ha="center", va="center", color="white")
    ax.add_patch(Rectangle((env.start[1], env.start[0]), 1, 1,
                           facecolor=RL_PALETTE["teal"], alpha=0.7))
    ax.text(env.start[1] + 0.5, env.start[0] + 0.5, "S", ha="center", va="center")
    ax.add_patch(Rectangle((env.goal[1], env.goal[0]), 1, 1,
                           facecolor=RL_PALETTE["gold"], alpha=0.7))
    ax.text(env.goal[1] + 0.5, env.goal[0] + 0.5, "G", ha="center", va="center")
    for k in range(env.cols + 1):
        ax.axvline(k, color="0.85", linewidth=0.5)
    for k in range(env.rows + 1):
        ax.axhline(k, color="0.85", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("DangerGridWorld")
    fig.tight_layout()
    fig.savefig(OUT / "fig_grid.pdf")
    plt.close(fig)


def fig_dfa():
    use_rl_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 6); ax.set_ylim(0, 3); ax.axis("off")
    c1 = Circle((1.5, 1.5), 0.6, edgecolor=RL_PALETTE["blue"], facecolor="white", linewidth=2)
    c2 = Circle((4.5, 1.5), 0.6, edgecolor=RL_PALETTE["red"], facecolor="white", linewidth=2)
    ax.add_patch(c1); ax.add_patch(c2)
    ax.text(1.5, 1.5, "$q_0$", ha="center", va="center")
    ax.text(4.5, 1.5, "$q_{bad}$", ha="center", va="center")
    ax.annotate("danger", xy=(3.9, 1.5), xytext=(2.1, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=RL_PALETTE["gray"]),
                ha="center", va="bottom")
    ax.annotate("", xy=(1.5, 0.85), xytext=(1.5, 0.2),
                arrowprops=dict(arrowstyle="-|>", color=RL_PALETTE["teal"],
                                connectionstyle="arc3,rad=0.3"))
    ax.text(0.7, 0.5, r"$\neg$danger", color=RL_PALETTE["teal"])
    ax.annotate("", xy=(4.5, 0.85), xytext=(4.5, 0.2),
                arrowprops=dict(arrowstyle="-|>", color=RL_PALETTE["gray"],
                                connectionstyle="arc3,rad=0.3"))
    ax.text(5.3, 0.5, r"$\cdot$ (sink)", color=RL_PALETTE["gray"])
    ax.set_title(r"DFA for safety property $G\,\neg$danger (q$_0$ accepting)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dfa.pdf")
    plt.close(fig)


def fig_safety_curves(seed: int = 0, n_episodes: int = 300):
    use_rl_style()
    h_s = train_shielded(n_episodes=n_episodes, seed=seed)
    h_u = train_unshielded(n_episodes=n_episodes, seed=seed)
    # Cumulative unsafe transitions.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(np.cumsum(h_u["unsafe"]), color=RL_PALETTE["red"], label="unshielded Q-learning")
    ax.plot(np.cumsum(h_s["unsafe"]), color=RL_PALETTE["teal"], label="shielded Q-learning")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative unsafe transitions")
    ax.set_title("Shield soundness in practice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_unsafe_curve.pdf")
    plt.close(fig)
    # Return curve.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    # Smooth a bit with a rolling mean.
    def rolling(x, w=20):
        x = np.asarray(x, dtype=float)
        return np.convolve(x, np.ones(w) / w, mode="valid")
    ax.plot(rolling(h_u["return"]), color=RL_PALETTE["gray"], label="unshielded")
    ax.plot(rolling(h_s["return"]), color=RL_PALETTE["blue"], label="shielded")
    ax.set_xlabel("Episode (smoothed)")
    ax.set_ylabel("Return")
    ax.set_title("Reward learning: shielding helps explore")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_return_curve.pdf")
    plt.close(fig)
    return h_s, h_u


def fig_pendulum_dr(seed: int = 0):
    use_rl_style()
    cfg = PendulumTrainConfig(n_epochs=25, steps_per_epoch=500)
    h_dr = train_pendulum_dr(seed=seed, randomise=True, cfg=cfg,
                             eval_lengths=(0.3, 0.6, 1.0, 1.4, 1.8))
    h_fx = train_pendulum_dr(seed=seed, randomise=False, cfg=cfg,
                             eval_lengths=(0.3, 0.6, 1.0, 1.4, 1.8))
    Ls = sorted(h_dr["eval"].keys())
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(Ls, [h_dr["eval"][L] for L in Ls], "o-",
            color=RL_PALETTE["blue"], label="DR train (lengths in [0.5,1.5])")
    ax.plot(Ls, [h_fx["eval"][L] for L in Ls], "s--",
            color=RL_PALETTE["orange"], label="fixed train (length=1.0)")
    ax.axvspan(0.5, 1.5, color=RL_PALETTE["teal"], alpha=0.1,
               label="DR train support")
    ax.set_xlabel("Eval pendulum length")
    ax.set_ylabel("Eval return (higher better)")
    ax.set_title("Robustness from domain randomisation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_pendulum_dr.pdf")
    plt.close(fig)


def main(seed: int = 0):
    OUT.mkdir(parents=True, exist_ok=True)
    fig_grid()
    fig_dfa()
    fig_safety_curves(seed=seed)
    fig_pendulum_dr(seed=seed)


if __name__ == "__main__":
    main()
