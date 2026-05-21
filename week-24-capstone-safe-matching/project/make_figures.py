"""Generate week-24 figures for notes.pdf.

Produces:
    figs/fig_pipeline.pdf         — schematic of the 3-phase pipeline
    figs/fig_iql_curves.pdf       — IQL q / v / pi loss curves
    figs/fig_ppo_lag_curves.pdf   — return, parity gap, lambda trajectory
    figs/fig_pareto.pdf           — Pareto frontier (parity vs SNIPS)
    figs/fig_calibration.pdf      — reliability diagram
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402

from .data import generate_dataset  # noqa: E402
from .offline_iql import iql_pretrain, IQLConfig  # noqa: E402
from .online_ppo_lag import ppo_lag_finetune, PPOLagConfig  # noqa: E402
from .audit import full_audit  # noqa: E402
from .train import train  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "figs"


def fig_pipeline():
    use_rl_style()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    blocks = [
        (0.5, "1. Logged data\n(epsilon-greedy)", RL_PALETTE["gray"]),
        (2.6, "2. IQL pretrain\n(in-support, biased)", RL_PALETTE["blue"]),
        (4.7, "3. PPO-Lagrangian\nfine-tune (fairness cost)", RL_PALETTE["teal"]),
        (6.8, "4. Audit\n(DP gap, EO, OPE)", RL_PALETTE["orange"]),
        (8.9, "5. Deploy\n(CLI / API)", RL_PALETTE["gold"]),
    ]
    for x, label, col in blocks:
        ax.add_patch(FancyBboxPatch((x, 1), 1.6, 1.2, boxstyle="round,pad=0.05",
                                    facecolor=col, alpha=0.6, edgecolor=col))
        ax.text(x + 0.8, 1.6, label, ha="center", va="center", fontsize=9)
    for x_start in (2.1, 4.2, 6.3, 8.4):
        ax.annotate("", xy=(x_start + 0.5, 1.6), xytext=(x_start, 1.6),
                    arrowprops=dict(arrowstyle="-|>", color="black"))
    ax.set_title("Capstone pipeline")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.pdf")
    plt.close(fig)


def fig_iql_and_ppo(seed: int = 0):
    use_rl_style()
    out = train(seed=seed,
                iql_cfg=IQLConfig(n_epochs=12),
                ppo_cfg=PPOLagConfig(n_updates=15, steps_per_update=384),
                n_candidates=600, n_jobs=30, n_interactions=2500)
    h = out["iql"]["history"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(h["q_loss"], color=RL_PALETTE["blue"], label="Q loss")
    axes[0].plot(h["v_loss"], color=RL_PALETTE["teal"], label="V loss")
    axes[0].plot(h["pi_loss"], color=RL_PALETTE["orange"], label=r"$\pi$ loss")
    axes[0].set_title("IQL training losses")
    axes[0].set_xlabel("Epoch"); axes[0].legend()
    p = out["ppo"]["history"]
    axes[1].plot(p["return"], color=RL_PALETTE["blue"])
    axes[1].set_title("Online return"); axes[1].set_xlabel("Update")
    axes[2].plot(p["parity"], color=RL_PALETTE["red"], label="parity gap")
    axes[2].plot(p["lambda"], color=RL_PALETTE["orange"], label=r"$\lambda$")
    axes[2].axhline(out["ppo"]["cfg"].cost_limit, color=RL_PALETTE["gray"], linestyle=":",
                    label="budget d")
    axes[2].set_title("Constraint + multiplier"); axes[2].set_xlabel("Update"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_iql_curves.pdf")
    plt.close(fig)
    # Also save the lambda/parity standalone.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(p["return"], color=RL_PALETTE["blue"], label="return")
    ax.plot(p["parity"], color=RL_PALETTE["red"], label="parity gap")
    ax.plot(p["lambda"], color=RL_PALETTE["orange"], label=r"$\lambda$")
    ax.axhline(out["ppo"]["cfg"].cost_limit, color=RL_PALETTE["gray"], linestyle=":")
    ax.set_xlabel("PPO-Lagrangian update")
    ax.legend(); ax.set_title("Online fine-tune")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ppo_lag_curves.pdf")
    plt.close(fig)
    return out


def fig_pareto(seed: int = 0):
    use_rl_style()
    pts = []
    for cl in (0.0, 0.03, 0.06, 0.12, 0.25):
        out = train(seed=seed, iql_cfg=IQLConfig(n_epochs=8),
                    ppo_cfg=PPOLagConfig(n_updates=10, steps_per_update=256, cost_limit=cl),
                    n_candidates=400, n_jobs=20, n_interactions=1600)
        pts.append({
            "cost_limit": cl,
            "parity_gap": out["audit_after"]["demographic_parity_gap"],
            "snips_value": out["audit_after"]["snips_value"],
        })
    parities = [p["parity_gap"] for p in pts]
    values = [p["snips_value"] for p in pts]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(parities, values, "o-", color=RL_PALETTE["blue"])
    for p in pts:
        ax.annotate(f"d={p['cost_limit']}", (p["parity_gap"], p["snips_value"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Demographic-parity gap (lower = fairer)")
    ax.set_ylabel("SNIPS off-policy value (higher = better)")
    ax.set_title("Fairness vs.\\ value Pareto frontier")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto.pdf")
    plt.close(fig)


def fig_calibration(out):
    use_rl_style()
    cal_b = out["audit_before"]["calibration"]
    cal_a = out["audit_after"]["calibration"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    ax.plot(cal_b["centres"], cal_b["empirical"], "o-",
            color=RL_PALETTE["orange"], label="before fine-tune")
    ax.plot(cal_a["centres"], cal_a["empirical"], "s-",
            color=RL_PALETTE["blue"], label="after fine-tune")
    ax.set_xlabel("Predicted P(recommend)")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Calibration / reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.pdf")
    plt.close(fig)


def main(seed: int = 0):
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    out = fig_iql_and_ppo(seed=seed)
    fig_calibration(out)
    fig_pareto(seed=seed)


if __name__ == "__main__":
    main()
