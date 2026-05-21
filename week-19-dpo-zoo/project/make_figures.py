"""Deterministic figures for week-19 notes.pdf."""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402
from common.seed import seed_everything  # noqa: E402

from .data import ToyPreferenceDataset  # noqa: E402
from .train import bakeoff  # noqa: E402


FIG_DIR = Path(__file__).resolve().parent.parent / "figs"


def main(seed: int = 0) -> None:
    use_rl_style()
    seed_everything(seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = ToyPreferenceDataset(n=64, seed=seed)
    results = bakeoff(
        ds.pairs, ds.sft_strings(),
        variants=("dpo", "ipo", "kto", "simpo", "orpo", "cpo"),
        sft_steps=120, pref_steps=80, seed=seed,
    )

    # Figure 1: loss curves.
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name, h in results.items():
        ax.plot(h["loss"], label=name.upper())
    ax.set_title("DPO zoo: training loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_loss.pdf"); plt.close(fig)

    # Figure 2: implicit reward margin.
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name, h in results.items():
        ax.plot(h["margin"], label=name.upper())
    ax.set_title("Reward margin (chosen vs.\\ rejected, higher is better)")
    ax.set_xlabel("step"); ax.set_ylabel("margin")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_margin.pdf"); plt.close(fig)

    # Figure 3: pairwise accuracy.
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name, h in results.items():
        ax.plot(h["acc"], label=name.upper())
    ax.set_title("Pairwise accuracy (chosen scored higher than rejected)")
    ax.set_xlabel("step"); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_acc.pdf"); plt.close(fig)

    # Figure 4: link-function comparison (the theory plot).
    x = np.linspace(-3, 3, 200)
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.plot(x, np.log1p(np.exp(-x)), label="DPO (logistic)", color=RL_PALETTE["blue"])
    ax.plot(x, (x - 5.0) ** 2 / 4.0, label="IPO (squared)", color=RL_PALETTE["teal"], ls="--")
    ax.plot(x, np.maximum(0, 1 - x), label="hinge (Slic-HF)", color=RL_PALETTE["orange"], ls=":")
    ax.set_xlabel("reward margin")
    ax.set_ylabel("loss")
    ax.set_title("Link functions in the Psi-PO family")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_links.pdf"); plt.close(fig)

    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
