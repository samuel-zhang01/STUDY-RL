"""Consistent matplotlib styling for STUDY-RL notebooks and figures."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

RL_PALETTE = {
    "blue":   "#1F3A5F",
    "teal":   "#2A9D8F",
    "orange": "#E76F51",
    "gold":   "#E9C46A",
    "gray":   "#4A4A4A",
    "red":    "#B23A48",
    "applyteal": "#0E7C7B",
}


def use_rl_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "axes.prop_cycle": plt.cycler("color", [
            RL_PALETTE["blue"], RL_PALETTE["teal"], RL_PALETTE["orange"],
            RL_PALETTE["gold"], RL_PALETTE["red"], RL_PALETTE["applyteal"],
        ]),
    })


def shaded(ax, x, ys, label=None, color=None, alpha=0.18, agg="mean"):
    """Plot mean (or median) ± std/IQR band across seeds.

    ys: array of shape (n_seeds, n_steps).
    """
    ys = np.asarray(ys)
    if agg == "mean":
        m = ys.mean(0)
        s = ys.std(0)
        lo, hi = m - s, m + s
    elif agg == "median":
        m = np.median(ys, 0)
        lo, hi = np.quantile(ys, 0.25, 0), np.quantile(ys, 0.75, 0)
    else:
        raise ValueError(agg)
    line, = ax.plot(x, m, color=color, label=label)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=alpha, linewidth=0)
    return line
