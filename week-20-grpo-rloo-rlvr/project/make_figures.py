"""Generate figures for ``notes.tex``.

Outputs (saved to ``figs/``):

- ``fig_grpo_accuracy.pdf``  — held-out greedy accuracy vs step.
- ``fig_grpo_reward.pdf``    — training-time reward mean vs step.
- ``fig_variance.pdf``       — variance comparison REINFORCE / RLOO / GRPO.

Deterministic given a seed. Total wall-clock ~30 s on CPU.
"""
from __future__ import annotations
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.plotting import RL_PALETTE, use_rl_style  # noqa: E402

from .train import train, GRPOConfig  # noqa: E402
from .solutions import exercise_2, exercise_3  # noqa: E402


FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIGS, exist_ok=True)


def _smooth(y, k=5):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def fig_training(seed: int = 0, num_steps: int = 400) -> None:
    use_rl_style()
    result = train(seed=seed, num_steps=num_steps, eval_every=25)
    hist = result["history"]

    # Accuracy.
    fig, ax = plt.subplots()
    ax.plot(hist["eval_step"], hist["eval_acc"], marker="o", color=RL_PALETTE["blue"], label="GRPO")
    ax.axhline(0.0, ls="--", color=RL_PALETTE["gray"], lw=0.8, label="random init")
    ax.set_xlabel("step")
    ax.set_ylabel("held-out greedy accuracy")
    ax.set_title("GRPO on toy arithmetic (verifiable reward)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_grpo_accuracy.pdf"))
    plt.close(fig)

    # Reward curve.
    fig, ax = plt.subplots()
    ax.plot(hist["step"], hist["reward_mean"], color=RL_PALETTE["teal"], alpha=0.35, lw=0.8)
    sm = _smooth(hist["reward_mean"], k=15)
    ax.plot(hist["step"][len(hist["step"]) - len(sm):], sm,
            color=RL_PALETTE["teal"], lw=1.8, label="reward (smoothed)")
    ax.set_xlabel("step")
    ax.set_ylabel("mean group reward")
    ax.set_title("Training-time verifier reward")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_grpo_reward.pdf"))
    plt.close(fig)


def fig_variance(seed: int = 0) -> None:
    """Compare estimator variance across group sizes."""
    use_rl_style()
    rng = np.random.default_rng(seed)
    Gs = [2, 4, 8, 16, 32]
    vrs_rein, vrs_rloo, vrs_grpo = [], [], []
    for G in Gs:
        # Bernoulli rewards with p=0.3 — verifiable-reward regime.
        r = (rng.uniform(size=(2000, G)) < 0.3).astype(float)
        vrs_rein.append(r.var(axis=-1).mean())
        vrs_rloo.append(np.var(np.stack([exercise_2(row) for row in r]), axis=-1).mean())
        vrs_grpo.append(np.var(exercise_3(r), axis=-1).mean())
    fig, ax = plt.subplots()
    ax.plot(Gs, vrs_rein, marker="o", label="REINFORCE (raw r)",  color=RL_PALETTE["orange"])
    ax.plot(Gs, vrs_rloo, marker="s", label="RLOO advantage",     color=RL_PALETTE["teal"])
    ax.plot(Gs, vrs_grpo, marker="^", label="GRPO advantage",     color=RL_PALETTE["blue"])
    ax.set_xlabel("group size G")
    ax.set_ylabel("Var of gradient coefficient")
    ax.set_xscale("log", base=2)
    ax.set_title("Variance vs group size (Bernoulli rewards, p=0.3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_variance.pdf"))
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    fig_training()
    fig_variance()
    print("figures saved to", FIGS)
