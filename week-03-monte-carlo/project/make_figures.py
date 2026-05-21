"""Figures for week-03 notes (Blackjack MC control results)."""
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

from .solutions import (
    exercise_3_blackjack_mc_control,
    exercise_6_compare_is_wis_variance,
)


FIG_DIR = os.path.join(WEEK, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def fig_blackjack_value() -> str:
    seed_everything(0)
    Q, pi, returns = exercise_3_blackjack_mc_control(n_episodes=40_000, eps=0.1, seed=0)
    # Build V(s) = max_a Q(s,a) for each (player_sum, dealer, usable_ace).
    V_no_ace = np.full((22 - 12, 10), np.nan)
    V_ace = np.full((22 - 12, 10), np.nan)
    for (obs, a), q in Q.items():
        s, d, u = obs
        if a != int(np.argmax([Q.get((obs, 0), -1e9), Q.get((obs, 1), -1e9)])):
            continue
        cell = V_ace if u == 1 else V_no_ace
        cell[s - 12, d - 1] = q
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, V, title in zip(axes, (V_no_ace, V_ace),
                            ("No usable ace", "Usable ace")):
        im = ax.imshow(V, origin="lower", aspect="auto", cmap="viridis",
                       extent=[1, 10, 12, 21])
        ax.set_xlabel("dealer showing"); ax.set_ylabel("player sum")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Blackjack — $V^\\star$ after 40k MC-control episodes")
    path = os.path.join(FIG_DIR, "fig_blackjack_value.pdf")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def fig_blackjack_learning_curve() -> str:
    seed_everything(0)
    _, _, returns = exercise_3_blackjack_mc_control(n_episodes=20_000, eps=0.1, seed=0)
    window = 500
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(smoothed, color=RL_PALETTE["blue"])
    ax.set_xlabel("episode"); ax.set_ylabel(f"average return (window={window})")
    ax.set_title("MC control on Blackjack")
    ax.axhline(y=-0.05, color=RL_PALETTE["orange"], ls="--", label="reference (~ -0.05)")
    ax.legend()
    path = os.path.join(FIG_DIR, "fig_blackjack_curve.pdf")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def fig_is_vs_wis() -> str:
    var_ois, var_wis = exercise_6_compare_is_wis_variance(n_trials=200, n_logs=200, seed=0)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(["OIS", "WIS"], [var_ois, var_wis],
           color=[RL_PALETTE["orange"], RL_PALETTE["blue"]])
    ax.set_ylabel("Var of estimator over 200 trials")
    ax.set_title("Ordinary IS vs Weighted IS variance")
    for i, v in enumerate([var_ois, var_wis]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    path = os.path.join(FIG_DIR, "fig_is_vs_wis.pdf")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def main():
    use_rl_style()
    for p in (fig_blackjack_value(), fig_blackjack_learning_curve(), fig_is_vs_wis()):
        print("wrote", p)


if __name__ == "__main__":
    main()
