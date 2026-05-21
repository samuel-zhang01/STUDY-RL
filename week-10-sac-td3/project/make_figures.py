"""Generate the SAC-on-Pendulum figure for the week-10 notes."""
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


def make_figures(seeds=(0, 1, 2), n_steps: int = 10000) -> None:
    use_rl_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    L = 50  # ~50 episodes per seed
    curves = []
    entropies = []
    for s in seeds:
        out = train(seed=s, n_steps=n_steps)
        r = np.array(out.episode_returns, dtype=float)
        if len(r) == 0:
            r = np.array([-1500.0])
        curves.append(_pad(_smooth(r, k=5), L))
        h = np.array(out.entropy_trace, dtype=float)
        if len(h) == 0:
            h = np.zeros(10)
        entropies.append(_pad(_smooth(h, k=50), 2000))
    curves = np.stack(curves, axis=0)
    entropies = np.stack(entropies, axis=0)

    fig, (ax_r, ax_h) = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=False)
    shaded(ax_r, np.arange(L), curves, label="SAC", color=RL_PALETTE["teal"], agg="median")
    ax_r.axhline(-200, color=RL_PALETTE["gray"], ls=":", label="solved (-200)")
    ax_r.set_ylabel("episode return")
    ax_r.set_title("SAC on Pendulum-v1")
    ax_r.legend(loc="lower right")

    shaded(ax_h, np.arange(2000), entropies,
           label="policy entropy", color=RL_PALETTE["orange"], agg="median")
    ax_h.axhline(-1.0, color=RL_PALETTE["gray"], ls=":",
                 label=r"target $\mathcal{H} = -|\mathcal{A}|$")
    ax_h.set_xlabel("gradient step")
    ax_h.set_ylabel(r"$-\log \pi$")
    ax_h.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "sac_pendulum.pdf")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    make_figures()
