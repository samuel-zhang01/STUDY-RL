"""Generate figures for week 21 ``notes.tex``.

Outputs:
- ``fig_hack_curves.pdf``    --- proxy vs true-quality divergence
- ``fig_regularizers.pdf``   --- effect of KL, length penalty, entropy
- ``fig_cai_compliance.pdf`` --- CAI compliance pre/post per step
- ``fig_gao_scaling.pdf``    --- toy reproduction of the Gao et al. 2022
                                 over-optimisation scaling law shape
"""
from __future__ import annotations
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.plotting import RL_PALETTE, use_rl_style  # noqa: E402

from .hack_lab import hack_train  # noqa: E402
from .cai import cai_train  # noqa: E402


FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIGS, exist_ok=True)


def fig_hack_curves(seed: int = 0, num_steps: int = 80):
    use_rl_style()
    res = hack_train(seed=seed, num_steps=num_steps, sft_warmup_steps=120,
                     batch=8, max_new_tokens=24)
    h = res["history"]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(h["step"], h["proxy"],         color=RL_PALETTE["orange"], label="proxy reward")
    ax2.plot(h["step"], h["true_quality"],  color=RL_PALETTE["blue"],   label="true quality")
    ax1.set_xlabel("step")
    ax1.set_ylabel("proxy reward", color=RL_PALETTE["orange"])
    ax2.set_ylabel("true quality (log-prob/char)", color=RL_PALETTE["blue"])
    ax2.grid(False)
    ax1.set_title("Reward hacking: proxy climbs while true quality drops")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_hack_curves.pdf"))
    plt.close(fig)


def fig_regularizers(seed: int = 0, num_steps: int = 60):
    use_rl_style()
    configs = {
        "no reg":         dict(),
        "KL (1.0)":       dict(kl_coef=1.0),
        "length penalty": dict(length_penalty=0.05),
    }
    fig, ax = plt.subplots()
    for (name, extra), color in zip(configs.items(), [RL_PALETTE["orange"], RL_PALETTE["teal"], RL_PALETTE["blue"]]):
        res = hack_train(seed=seed, num_steps=num_steps, sft_warmup_steps=120,
                         batch=8, max_new_tokens=24, **extra)
        ax.plot(res["history"]["step"], res["history"]["true_quality"],
                label=name, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("true quality (log-prob/char)")
    ax.set_title("Mitigations: KL anchor and length penalty preserve true quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_regularizers.pdf"))
    plt.close(fig)


def fig_cai_compliance(seed: int = 0, num_steps: int = 40):
    use_rl_style()
    res = cai_train(seed=seed, num_steps=num_steps, sft_warmup_steps=100,
                    batch=4, max_new_tokens=24)
    h = res["history"]
    fig, ax = plt.subplots()
    ax.plot(h["step"], h["compliance_pre"],  label="pre-revision", color=RL_PALETTE["orange"])
    ax.plot(h["step"], h["compliance_post"], label="post-revision (chosen)", color=RL_PALETTE["teal"])
    ax.set_xlabel("step")
    ax.set_ylabel("mean compliance score (0-5)")
    ax.set_title("Constitutional AI: revision improves compliance per step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_cai_compliance.pdf"))
    plt.close(fig)


def fig_gao_scaling(seed: int = 0):
    """Reproduce the *shape* of Gao et al. 2022's reward-overoptimisation
    scaling law on a toy.

    The Gao paper models true reward as
        R_true(kl) = R_proxy(kl) - phi(kl)
    where phi(kl) grows as kl increases beyond a threshold. We use a
    quadratic phi for the toy.
    """
    use_rl_style()
    rng = np.random.default_rng(seed)
    kl_grid = np.linspace(0, 25, 200)
    # Proxy improves with sqrt of KL (returns diminish).
    r_proxy = 2.0 * np.sqrt(kl_grid)
    # True reward is proxy minus a quadratic penalty representing Goodhart
    # drift; the optimum is at moderate KL.
    r_true = r_proxy - 0.04 * kl_grid ** 2
    fig, ax = plt.subplots()
    ax.plot(kl_grid, r_proxy, label="proxy reward", color=RL_PALETTE["orange"])
    ax.plot(kl_grid, r_true,  label="true reward",  color=RL_PALETTE["blue"])
    ax.axvline(kl_grid[np.argmax(r_true)], ls="--", color=RL_PALETTE["gray"],
               lw=0.8, label="argmax true")
    ax.set_xlabel(r"KL($\pi_\theta \,\|\, \pi_\text{ref}$)")
    ax.set_ylabel("reward")
    ax.set_title("Gao 2022 over-optimisation law (toy reproduction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_gao_scaling.pdf"))
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    fig_hack_curves()
    fig_regularizers()
    fig_cai_compliance()
    fig_gao_scaling()
    print("figures saved to", FIGS)
