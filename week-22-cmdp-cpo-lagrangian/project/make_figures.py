"""Generate figures used in notes.pdf. Deterministic given the seed.

Produces:
    figs/fig_return_cost.pdf      — return + cost curves with the budget line
    figs/fig_lambda.pdf           — Lagrange multiplier trajectory
    figs/fig_safe_vs_unsafe.pdf   — strict vs lax cost-limit comparison
    figs/fig_lavagrid.pdf         — diagram of the gridworld
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402

from .env import LavaGridWorld  # noqa: E402
from .train import train  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "figs"


def _ensure_out():
    OUT.mkdir(parents=True, exist_ok=True)


def fig_return_cost(seed: int = 0):
    use_rl_style()
    h = train(seed=seed, n_updates=80, steps_per_update=512, cost_limit=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    x = np.arange(len(h["return"]))
    axes[0].plot(x, h["return"], color=RL_PALETTE["blue"], label="episodic return")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Return $J_r$")
    axes[0].set_title("Reward learning curve")
    axes[0].legend()
    axes[1].plot(x, h["cost"], color=RL_PALETTE["red"], label="discounted cost $J_c$")
    axes[1].axhline(1.5, color=RL_PALETTE["gray"], linestyle=":", label="budget $d=1.5$")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Cost $J_c$")
    axes[1].set_title("Constraint satisfaction")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_return_cost.pdf")
    plt.close(fig)
    return h


def fig_lambda(history):
    use_rl_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(history["lambda"], color=RL_PALETTE["orange"])
    ax.set_xlabel("Update")
    ax.set_ylabel(r"Lagrange multiplier $\lambda$")
    ax.set_title("Dual variable trajectory")
    fig.tight_layout()
    fig.savefig(OUT / "fig_lambda.pdf")
    plt.close(fig)


def fig_safe_vs_unsafe(seed: int = 0):
    use_rl_style()
    h_safe = train(seed=seed, n_updates=60, steps_per_update=512, cost_limit=0.5)
    h_unsafe = train(seed=seed, n_updates=60, steps_per_update=512, cost_limit=1e6)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].plot(h_unsafe["return"], color=RL_PALETTE["gray"], label="unconstrained")
    axes[0].plot(h_safe["return"], color=RL_PALETTE["blue"], label=r"PPO-Lag, $d=0.5$")
    axes[0].set_title("Return: safe vs unsafe")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[1].plot(h_unsafe["cost"], color=RL_PALETTE["gray"], label="unconstrained")
    axes[1].plot(h_safe["cost"], color=RL_PALETTE["red"], label=r"PPO-Lag, $d=0.5$")
    axes[1].axhline(0.5, color=RL_PALETTE["gold"], linestyle=":")
    axes[1].set_title("Cost: safe vs unsafe")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Cost $J_c$")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_safe_vs_unsafe.pdf")
    plt.close(fig)


def fig_lavagrid():
    use_rl_style()
    env = LavaGridWorld()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    for (r, c) in env.lava:
        ax.add_patch(Rectangle((c, r), 1, 1,
                               facecolor=RL_PALETTE["red"], alpha=0.7))
    ax.add_patch(Rectangle((env.start[1], env.start[0]), 1, 1,
                           facecolor=RL_PALETTE["teal"], alpha=0.7))
    ax.add_patch(Rectangle((env.goal[1], env.goal[0]), 1, 1,
                           facecolor=RL_PALETTE["gold"], alpha=0.7))
    ax.text(env.start[1] + 0.5, env.start[0] + 0.5, "S", ha="center", va="center")
    ax.text(env.goal[1] + 0.5, env.goal[0] + 0.5, "G", ha="center", va="center")
    for (r, c) in env.lava:
        ax.text(c + 0.5, r + 0.5, "L", ha="center", va="center", color="white")
    for k in range(env.cols + 1):
        ax.axvline(k, color="0.85", linewidth=0.5)
    for k in range(env.rows + 1):
        ax.axhline(k, color="0.85", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("LavaGridWorld (S=start, G=goal, L=lava cost 1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_lavagrid.pdf")
    plt.close(fig)


def main(seed: int = 0):
    _ensure_out()
    fig_lavagrid()
    h = fig_return_cost(seed=seed)
    fig_lambda(h)
    fig_safe_vs_unsafe(seed=seed)


if __name__ == "__main__":
    main()
