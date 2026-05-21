"""Figures for week 13's notes.tex.

Outputs:
    fig_az_winrate.pdf   - AlphaZero winrate vs random across iterations
    fig_az_losses.pdf    - Policy and value losses across training
    fig_coord_game.pdf   - MAPPO learning curve on the coordination game
    fig_simple_spread.pdf - MAPPO learning curve on simple_spread (noisy)
    fig_puct.pdf         - PUCT bonus illustration
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
ROOT = WEEK.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WEEK))

from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402
from common.seed import seed_everything  # noqa: E402

import project as P  # noqa: E402, F401
from project.train import train_alphazero  # noqa: E402
from project.mappo import train_mappo  # noqa: E402
from project.coord_game import CoordGame  # noqa: E402

FIGS = WEEK / "figs"
FIGS.mkdir(exist_ok=True)


def fig_az_winrate(seed: int = 0) -> Path:
    use_rl_style()
    seed_everything(seed)
    out = train_alphazero(seed=seed, n_iters=12, n_episodes_per_iter=5,
                           n_simulations=25)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(np.arange(1, len(out["winrates"]) + 1), out["winrates"],
            "o-", color=RL_PALETTE["teal"], label="winrate vs random")
    ax.axhline(0.5, color=RL_PALETTE["gray"], linestyle="--",
               label="random baseline")
    ax.set_xlabel("iteration")
    ax.set_ylabel("winrate (wins + 0.5 draws) / games")
    ax.set_ylim(0, 1.05)
    ax.set_title("AlphaZero on Tic-Tac-Toe: winrate vs.\\ random opponent")
    ax.legend()
    out_path = FIGS / "fig_az_winrate.pdf"
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    # Save losses too while we're here
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
    ax2.plot(out["policy_losses"], color=RL_PALETTE["blue"], label="policy")
    ax2.plot(out["value_losses"], color=RL_PALETTE["orange"], label="value")
    ax2.set_xlabel("update step"); ax2.set_ylabel("loss")
    ax2.set_title("AlphaZero training losses")
    ax2.legend()
    out2 = FIGS / "fig_az_losses.pdf"
    fig2.tight_layout(); fig2.savefig(out2); plt.close(fig2)
    return out_path


def fig_coord_game(seed: int = 0) -> Path:
    use_rl_style()
    seed_everything(seed)
    env = CoordGame(n_actions=3, n_agents=2, seed=seed)
    out = train_mappo(env=env, n_episodes=200, seed=seed, lr=1e-3,
                      update_epochs=4)
    r = out["returns"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(r, alpha=0.3, color=RL_PALETTE["teal"])
    # Moving average
    win = 20
    ma = np.convolve(r, np.ones(win) / win, mode="valid")
    ax.plot(np.arange(win - 1, len(r)), ma, color=RL_PALETTE["teal"],
            label=f"{win}-ep moving avg")
    ax.axhline(10.0, color=RL_PALETTE["gray"], linestyle="--",
               label="optimal (perfect coordination)")
    ax.set_xlabel("episode"); ax.set_ylabel("team return")
    ax.set_title("MAPPO on 2-agent matrix coordination game")
    ax.legend()
    out_path = FIGS / "fig_coord_game.pdf"
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def fig_simple_spread(seed: int = 0) -> Path:
    use_rl_style()
    seed_everything(seed)
    out = train_mappo(n_episodes=120, seed=seed, lr=3e-4, update_epochs=4)
    r = out["returns"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(r, alpha=0.3, color=RL_PALETTE["orange"])
    win = 15
    ma = np.convolve(r, np.ones(win) / win, mode="valid")
    ax.plot(np.arange(win - 1, len(r)), ma, color=RL_PALETTE["orange"],
            label=f"{win}-ep moving avg")
    ax.set_xlabel("episode"); ax.set_ylabel("team return")
    ax.set_title("MAPPO on 3-agent SimpleSpread (cooperative coverage)")
    ax.legend()
    out_path = FIGS / "fig_simple_spread.pdf"
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def fig_puct(seed: int = 0) -> Path:
    use_rl_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    # PUCT = -q + c_puct * P * sqrt(N) / (1 + n)
    ns = np.arange(1, 60)
    for c, lbl, col in (
        (0.5, "low prior $P=0.05$", RL_PALETTE["blue"]),
        (0.25, "medium prior $P=0.25$", RL_PALETTE["teal"]),
        (0.5, "high prior $P=0.50$", RL_PALETTE["orange"]),
    ):
        # Re-use ns for parent visits and child visits varying together
        pass
    # Easier: plot bonus for fixed parent visits N=100, varying child visits.
    N = 100
    c_puct = 1.5
    n = np.arange(1, 60)
    for P_val, col in zip([0.05, 0.25, 0.5],
                          [RL_PALETTE["blue"], RL_PALETTE["teal"], RL_PALETTE["orange"]]):
        bonus = c_puct * P_val * np.sqrt(N) / (1 + n)
        ax.plot(n, bonus, color=col, label=f"prior $P = {P_val}$")
    ax.set_xlabel("child visits $n$")
    ax.set_ylabel("PUCT exploration bonus")
    ax.set_title(r"PUCT bonus = $c \cdot P \cdot \sqrt{N}/(1 + n)$, $N=100$, $c=1.5$")
    ax.legend()
    out_path = FIGS / "fig_puct.pdf"
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def main() -> None:
    print(fig_az_winrate())
    print(fig_coord_game())
    print(fig_simple_spread())
    print(fig_puct())


if __name__ == "__main__":
    main()
