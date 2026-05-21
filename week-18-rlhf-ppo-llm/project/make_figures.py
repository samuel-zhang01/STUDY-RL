"""Deterministic figures for notes.pdf — runs in <60 s on CPU."""
from __future__ import annotations
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402
from common.seed import seed_everything  # noqa: E402

from .data import ToyPreferenceDataset, heuristic_reward  # noqa: E402
from .model import CharTransformer, RewardModel, TransformerConfig  # noqa: E402
from .sft import sft_train  # noqa: E402
from .rm import rm_train  # noqa: E402
from .ppo import ppo_train, PPOConfig  # noqa: E402


FIG_DIR = Path(__file__).resolve().parent.parent / "figs"


def main(seed: int = 0) -> None:
    use_rl_style()
    seed_everything(seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = ToyPreferenceDataset(n=64, seed=seed)
    cfg = TransformerConfig()
    policy = CharTransformer(cfg)
    rm_back = CharTransformer(cfg)
    rm = RewardModel(rm_back)

    sft_losses = sft_train(policy, ds.sft_strings(), steps=150, seed=seed)
    rm_losses, rm_accs = rm_train(rm, ds.pairs, steps=150, seed=seed)
    history, _, _ = ppo_train(
        policy, rm, ds.prompts(),
        cfg=PPOConfig(steps=30, batch_size=4, max_new_tokens=16, seed=seed),
    )

    # Figure 1: SFT and RM losses.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axes[0].plot(sft_losses, color=RL_PALETTE["blue"])
    axes[0].set_title("SFT loss"); axes[0].set_xlabel("step"); axes[0].set_ylabel("nats/tok")
    axes[1].plot(rm_losses, color=RL_PALETTE["teal"], label="loss")
    ax2 = axes[1].twinx(); ax2.plot(rm_accs, color=RL_PALETTE["orange"], label="acc")
    ax2.set_ylim(0, 1.05); ax2.grid(False)
    axes[1].set_title("RM Bradley--Terry loss / acc")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sft_rm.pdf"); plt.close(fig)

    # Figure 2: PPO reward + KL.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axes[0].plot(history["reward"], color=RL_PALETTE["blue"])
    axes[0].set_title("PPO: mean RM score"); axes[0].set_xlabel("step")
    axes[1].plot(history["kl"], color=RL_PALETTE["red"], label="approx KL")
    axes[1].plot(history["beta"], color=RL_PALETTE["gold"], label=r"$\beta$")
    axes[1].set_title("KL controller")
    axes[1].set_xlabel("step"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ppo_curves.pdf"); plt.close(fig)

    # Figure 3: reward-model overoptimisation cartoon.
    # We fit a quadratic of "gold" reward vs proxy reward as KL grows
    # to evoke Gao 2022 — using synthetic data (this is illustrative).
    kl = np.linspace(0, 6, 80)
    proxy = 1 - np.exp(-0.7 * kl)
    gold = proxy * (1 - 0.06 * kl) - 0.02 * kl
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(kl, proxy, color=RL_PALETTE["teal"], label="proxy (learned RM)")
    ax.plot(kl, gold, color=RL_PALETTE["red"], label="gold (true objective)")
    ax.axvline(kl[np.argmax(gold)], color=RL_PALETTE["gray"], ls=":", lw=1)
    ax.set_xlabel(r"KL$(\pi_\theta\,\|\,\pi_{\text{ref}})$ [nats]")
    ax.set_ylabel("expected reward")
    ax.set_title("Reward-model overoptimisation (Gao et al. 2023, schematic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_overoptimisation.pdf"); plt.close(fig)

    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
