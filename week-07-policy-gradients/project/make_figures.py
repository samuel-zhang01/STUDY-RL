"""Generate the two figures referenced by notes.tex:
  figs/learning_curve.pdf       (with vs without baseline)
  figs/variance_reduction.pdf   (estimator variance over training)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import use_rl_style, RL_PALETTE, shaded
from .train import train


FIG_DIR = Path(__file__).resolve().parent.parent / "figs"


def _smooth(x: np.ndarray, k: int = 20) -> np.ndarray:
    if len(x) < k:
        return x
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    pad = np.full(k - 1, c[0])
    return np.concatenate([pad, c])


def make_figures(seeds=(0, 1, 2), n_episodes: int = 400) -> None:
    use_rl_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    curves_with: list[np.ndarray] = []
    curves_without: list[np.ndarray] = []
    vars_with: list[np.ndarray] = []
    vars_without: list[np.ndarray] = []

    for s in seeds:
        out_b = train(seed=s, n_episodes=n_episodes, use_baseline=True)
        out_n = train(seed=s, n_episodes=n_episodes, use_baseline=False)
        curves_with.append(_smooth(np.array(out_b.returns)))
        curves_without.append(_smooth(np.array(out_n.returns)))
        vars_with.append(_smooth(np.array(out_b.estimator_variances)))
        vars_without.append(_smooth(np.array(out_n.estimator_variances)))

    x = np.arange(n_episodes)

    # ---- learning curve ----
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    shaded(ax, x, np.stack(curves_without), label="REINFORCE",
           color=RL_PALETTE["blue"])
    shaded(ax, x, np.stack(curves_with), label=r"REINFORCE + $V_\phi$",
           color=RL_PALETTE["teal"])
    ax.axhline(195, color=RL_PALETTE["gray"], linestyle=":", linewidth=1,
               label="solved threshold")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return (CartPole-v1)")
    ax.set_title("Learning curve: variance-reduced PG converges faster")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curve.pdf")
    plt.close(fig)

    # ---- variance reduction ----
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    shaded(ax, x, np.stack(vars_without), label="REINFORCE (no baseline)",
           color=RL_PALETTE["blue"])
    shaded(ax, x, np.stack(vars_with), label=r"REINFORCE + $V_\phi$",
           color=RL_PALETTE["teal"])
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("variance of log-probability × advantage terms (log scale)")
    ax.set_title("Within-rollout loss-term variance diagnostic")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "variance_reduction.pdf")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    make_figures()
