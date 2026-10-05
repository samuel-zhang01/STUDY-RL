"""Generate the two figures for the week-09 notes:
  figs/ppo_walker.pdf            -- PPO BipedalWalker learning curve
  figs/details_ablation.pdf      -- ablate three of the 37 details
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import use_rl_style, RL_PALETTE, shaded

from .train import train


FIG_DIR = Path(__file__).resolve().parent.parent / "figs"


def _smooth(x: np.ndarray, k: int = 10) -> np.ndarray:
    if len(x) < k:
        return np.asarray(x)
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    pad = np.full(k - 1, c[0])
    return np.concatenate([pad, c])


def _pad(x: np.ndarray, L: int) -> np.ndarray:
    if len(x) >= L:
        return x[:L]
    return np.concatenate([x, np.full(L - len(x), x[-1] if len(x) else 0.0)])


def make_figures(seeds=(0, 1, 2), n_updates: int = 60) -> None:
    use_rl_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    curves = []
    for s in seeds:
        out = train(seed=s, n_updates=n_updates)
        r = np.array(out.episode_returns, dtype=float)
        if len(r) == 0:
            raise ValueError("No completed episodes available for the learning curve")
        curves.append(_smooth(r, k=10))
    L = min(len(curve) for curve in curves)
    curves = np.stack([curve[:L] for curve in curves], axis=0)
    x = np.arange(L)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    shaded(ax, x, curves, label="PPO (implemented details)", color=RL_PALETTE["teal"], agg="median")
    ax.set_xlabel("episode index")
    ax.set_ylabel("episode return (BipedalWalker-v3)")
    ax.set_title("PPO on BipedalWalker-v3")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ppo_walker.pdf")
    plt.close(fig)

    # Ablation: full vs missing details.
    variants = {
        "full": dict(),
        "no adv-norm": dict(use_advantage_norm=False),
        "no value-clip": dict(use_value_clip=False),
        "no LR anneal": dict(anneal_lr=False),
    }
    finals: dict[str, list[float]] = {k: [] for k in variants}
    for name, kw in variants.items():
        for s in seeds:
            out = train(seed=s, n_updates=max(30, n_updates // 2), **kw)
            r = np.array(out.episode_returns[-20:]) if out.episode_returns else np.array([0.0])
            finals[name].append(float(r.mean()))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    names = list(variants.keys())
    means = [np.mean(finals[n]) for n in names]
    stds = [np.std(finals[n]) for n in names]
    colors = [RL_PALETTE["teal"], RL_PALETTE["blue"], RL_PALETTE["orange"], RL_PALETTE["gold"]]
    ax.bar(names, means, yerr=stds, color=colors, capsize=4)
    ax.set_ylabel("final mean return (last 20 episodes)")
    ax.set_title("PPO implementation-detail ablation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "details_ablation.pdf")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    make_figures()
