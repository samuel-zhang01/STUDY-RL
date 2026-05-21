"""Figures for notes.tex.

Outputs:
    fig_tt_loss.pdf       - two-tower contrastive training loss
    fig_ndcg_compare.pdf  - NDCG@k bar chart: random vs two-tower vs reranker
    fig_fairness.pdf      - DP gap, EO gap, exposure pre vs post
    fig_calibration.pdf   - reliability diagrams for raw / Platt / Iso / Beta
    fig_position_bias.pdf - position-bias curve and click-through cartoon

Deterministic on seed 0.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
ROOT = WEEK.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WEEK))

from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402
from common.seed import seed_everything  # noqa: E402

import project as P  # noqa: E402, F401
from project.env import ATSDataset  # noqa: E402
from project.two_tower import train_two_tower, all_scores  # noqa: E402
from project.train import train, ndcg_at_k  # noqa: E402
from project.cltr import position_bias  # noqa: E402
from project.calibration import (  # noqa: E402
    PlattCalibrator, IsotonicCalibrator, BetaCalibrator,
    reliability_curve, ece,
)

FIGS = WEEK / "figs"
FIGS.mkdir(exist_ok=True)


def fig_tt_loss(seed: int = 0) -> Path:
    use_rl_style()
    seed_everything(seed)
    ds = ATSDataset(seed=seed)
    _, losses = train_two_tower(ds, epochs=60, seed=seed)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(losses, color=RL_PALETTE["teal"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("symmetric InfoNCE loss")
    ax.set_title("Two-tower contrastive training (200 candidates, 50 jobs)")
    out = FIGS / "fig_tt_loss.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_ndcg_compare(seed: int = 0) -> Path:
    use_rl_style()
    seed_everything(seed)
    out = train(seed=seed, epochs=40)
    ds = ATSDataset(seed=seed)
    rel = ds.true_match_matrix()
    # Random ranking baseline.
    rng = np.random.default_rng(seed + 13)
    rand_scores = rng.uniform(0, 1, size=rel.shape)
    rand_ndcg = ndcg_at_k(rand_scores, rel, k=10)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    names = ["random", "two-tower", "two-tower + LinTS"]
    vals = [rand_ndcg, out.metrics["tt_ndcg@10"], out.metrics["rerank_ndcg@10"]]
    colors = [RL_PALETTE["gray"], RL_PALETTE["teal"], RL_PALETTE["orange"]]
    ax.bar(names, vals, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Ranking quality: retrieval vs.\\ retrieval + bandit reranker")
    out_path = FIGS / "fig_ndcg_compare.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_fairness(seed: int = 0) -> Path:
    use_rl_style()
    seed_everything(seed)
    out = train(seed=seed, epochs=40)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    # left: gaps
    ax1.bar(["DP gap pre", "DP gap post", "EO gap pre", "EO gap post"], [
        out.fairness_pre["dp_gap"], out.fairness_post["dp_gap"],
        out.fairness_pre["eo_gap"], out.fairness_post["eo_gap"],
    ], color=[RL_PALETTE["blue"], RL_PALETTE["teal"],
              RL_PALETTE["red"], RL_PALETTE["orange"]])
    ax1.set_title("Fairness gaps")
    ax1.set_ylabel("|gap|")
    ax1.tick_params(axis="x", rotation=15)

    # right: exposure
    ax2.bar(["exp F (pre)", "exp M (pre)", "exp F (post)", "exp M (post)"], [
        out.fairness_pre["exposure_F"], out.fairness_pre["exposure_M"],
        out.fairness_post["exposure_F"], out.fairness_post["exposure_M"],
    ], color=[RL_PALETTE["blue"], RL_PALETTE["teal"],
              RL_PALETTE["red"], RL_PALETTE["orange"]])
    ax2.set_title("Position-bias-weighted exposure")
    ax2.set_ylabel("avg exposure / candidate / job")
    ax2.tick_params(axis="x", rotation=15)
    out_path = FIGS / "fig_fairness.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_calibration(seed: int = 0) -> Path:
    use_rl_style()
    seed_everything(seed)
    # Synthetic over-confident scores (raw ~ Uniform[0,1], P(y=1) = raw^2).
    rng = np.random.default_rng(seed)
    n = 4000
    raw = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < raw ** 2).astype(int)
    half = n // 2
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot([0, 1], [0, 1], "--", color=RL_PALETTE["gray"], label="perfect")
    palette = [RL_PALETTE["blue"], RL_PALETTE["teal"], RL_PALETTE["orange"], RL_PALETTE["red"]]
    cur = raw[half:]
    cur_lbl = labels[half:]
    centres, acc, cnt = reliability_curve(cur, cur_lbl)
    ax.plot(centres, acc, "o-", color=palette[0], label=f"raw  (ECE={ece(cur, cur_lbl):.3f})")
    for i, (name, klass) in enumerate(
        [("Platt", PlattCalibrator), ("Isotonic", IsotonicCalibrator), ("Beta", BetaCalibrator)],
        start=1,
    ):
        cal = klass().fit(raw[:half], labels[:half])
        p = cal(raw[half:])
        centres, acc, cnt = reliability_curve(p, labels[half:])
        ax.plot(centres, acc, "o-", color=palette[i],
                label=f"{name} (ECE={ece(p, labels[half:]):.3f})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagrams (synthetic over-confident scores)")
    ax.legend(loc="upper left")
    out_path = FIGS / "fig_calibration.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_position_bias(seed: int = 0) -> Path:
    use_rl_style()
    K = 20
    pb1 = position_bias(K, eta=1.0)
    pb05 = position_bias(K, eta=0.5)
    pb15 = position_bias(K, eta=1.5)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(1, K + 1)
    ax.plot(x, pb1, "-o", label=r"$\eta=1.0$", color=RL_PALETTE["blue"])
    ax.plot(x, pb05, "-s", label=r"$\eta=0.5$", color=RL_PALETTE["teal"])
    ax.plot(x, pb15, "-^", label=r"$\eta=1.5$", color=RL_PALETTE["orange"])
    ax.set_xlabel("position $k$")
    ax.set_ylabel(r"examination prob.\ $p_k$")
    ax.set_title("Position-bias model (Joachims et al.\\ WSDM 2017)")
    ax.legend()
    out_path = FIGS / "fig_position_bias.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    print(fig_tt_loss())
    print(fig_ndcg_compare())
    print(fig_fairness())
    print(fig_calibration())
    print(fig_position_bias())


if __name__ == "__main__":
    main()
