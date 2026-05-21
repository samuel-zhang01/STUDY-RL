"""Generate the two figures for the week-08 notes:
  figs/a2c_lander.pdf    -- A2C+GAE LunarLanderContinuous training curve
  figs/lambda_sweep.pdf  -- final return vs lambda for fixed compute
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


def make_figures(seeds=(0, 1, 2), n_updates: int = 150) -> None:
    use_rl_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- A2C learning curve ---
    L = 800  # pad / trim to a common length
    curves = []
    for s in seeds:
        out = train(seed=s, n_updates=n_updates)
        r = np.array(out.episode_returns, dtype=float)
        if len(r) == 0:
            r = np.array([0.0])
        sm = _smooth(r, k=10)
        # pad/trim to same length so we can stack
        if len(sm) < L:
            sm = np.concatenate([sm, np.full(L - len(sm), sm[-1])])
        else:
            sm = sm[:L]
        curves.append(sm)
    curves = np.stack(curves, axis=0)
    x = np.arange(L)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    shaded(ax, x, curves, label="A2C+GAE", color=RL_PALETTE["teal"], agg="median")
    ax.set_xlabel("episode index")
    ax.set_ylabel("episode return (LunarLanderContinuous-v3)")
    ax.set_title("A2C+GAE on LunarLanderContinuous-v3")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "a2c_lander.pdf")
    plt.close(fig)

    # --- lambda sweep ---
    lams = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    finals: list[list[float]] = []
    for lam in lams:
        seed_finals = []
        for s in seeds:
            out = train(seed=s, n_updates=max(60, n_updates // 2), lam=lam)
            r = np.array(out.episode_returns[-30:]) if len(out.episode_returns) > 0 else np.array([0.0])
            seed_finals.append(float(r.mean()))
        finals.append(seed_finals)
    finals_arr = np.array(finals)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    means = finals_arr.mean(axis=1)
    stds = finals_arr.std(axis=1)
    ax.errorbar(lams, means, yerr=stds, color=RL_PALETTE["blue"],
                marker="o", capsize=3, label=r"final return $\pm$ std")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("final episode return (mean of last 30)")
    ax.set_title("GAE $\\lambda$ sweep (fixed compute)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lambda_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    make_figures()
