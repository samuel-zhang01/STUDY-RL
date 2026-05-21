"""Figures for week-02 notes."""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK = os.path.dirname(HERE)
REPO = os.path.dirname(WEEK)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402
from common.seed import seed_everything  # noqa: E402
from common.gridworld import GridWorld  # noqa: E402
from common.tabular import value_iteration, policy_iteration  # noqa: E402

FIG_DIR = os.path.join(WEEK, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def fig_convergence() -> str:
    seed_everything(0)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for gamma, colour in [(0.5, RL_PALETTE["orange"]),
                          (0.9, RL_PALETTE["teal"]),
                          (0.99, RL_PALETTE["blue"])]:
        env = GridWorld(rows=6, cols=6, start=(0, 0), goals=((5, 5),))
        _, _, deltas = value_iteration(env, gamma=gamma, tol=1e-12, max_iter=400)
        ax.semilogy(deltas, color=colour, label=f"$\\gamma={gamma}$")
    ax.set_xlabel("sweep")
    ax.set_ylabel(r"$\|V_{k+1} - V_k\|_\infty$ (log)")
    ax.set_title("Value-iteration sup-norm contraction on a 6x6 GridWorld")
    ax.legend()
    path = os.path.join(FIG_DIR, "fig_convergence.pdf")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def fig_value_heatmap() -> str:
    env = GridWorld(rows=6, cols=6, start=(0, 0),
                    goals=((5, 5),), pits=((2, 2), (3, 4)))
    V, pi, _ = value_iteration(env, gamma=0.95)
    grid = V.reshape(env.rows, env.cols)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(grid, cmap="viridis")
    ax.set_title("Optimal $V^\\star$ on a 6x6 GridWorld\n(goal=(5,5), pits=(2,2),(3,4))")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # overlay arrows for the greedy policy
    arrows = ["U", "R", "D", "L"]
    for s in range(env.nS):
        r, c = env.idx2s(s)
        if (r, c) in env.goals or (r, c) in env.pits:
            continue
        ax.text(c, r, arrows[int(pi[s])], ha="center", va="center", color="white",
                fontsize=10)
    path = os.path.join(FIG_DIR, "fig_value_heatmap.pdf")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def fig_pi_vs_vi() -> str:
    """Compare PI vs VI sweep counts as the discount factor varies."""
    gammas = np.linspace(0.5, 0.995, 12)
    vi_sweeps = []
    pi_sweeps = []
    for g in gammas:
        env = GridWorld(rows=5, cols=5, start=(0, 0), goals=((4, 4),))
        _, _, d = value_iteration(env, gamma=float(g), tol=1e-10, max_iter=2000)
        vi_sweeps.append(len(d))
        env = GridWorld(rows=5, cols=5, start=(0, 0), goals=((4, 4),))
        _, _, n = policy_iteration(env, gamma=float(g), max_iter=200)
        pi_sweeps.append(n)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(gammas, vi_sweeps, "o-", color=RL_PALETTE["blue"], label="Value iteration")
    ax.plot(gammas, pi_sweeps, "s-", color=RL_PALETTE["orange"], label="Policy iteration")
    ax.set_xlabel("$\\gamma$"); ax.set_ylabel("sweeps to convergence")
    ax.set_yscale("log")
    ax.set_title("VI vs PI on a 5x5 GridWorld")
    ax.legend()
    path = os.path.join(FIG_DIR, "fig_pi_vs_vi.pdf")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def main() -> None:
    use_rl_style()
    for p in (fig_convergence(), fig_value_heatmap(), fig_pi_vs_vi()):
        print("wrote", p)


if __name__ == "__main__":
    main()
