"""Generate figures used in week-01 notes.

Run with ``python -m week_01_mdps_bandits.project.make_figures`` or directly.
Outputs go to ``figs/*.pdf`` (next to ``notes.tex``).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# allow running as a plain script
HERE = os.path.dirname(os.path.abspath(__file__))
WEEK = os.path.dirname(HERE)
REPO = os.path.dirname(WEEK)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from common.plotting import use_rl_style, RL_PALETTE, shaded  # noqa: E402
from common.seed import seed_everything  # noqa: E402

from .train import run_bernoulli_experiment, run_contextual_experiment  # noqa: E402
from .env import SyntheticLeadBrokerBandit  # noqa: E402


FIG_DIR = os.path.join(WEEK, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def fig_bernoulli_bakeoff() -> str:
    seed_everything(0)
    means = np.array([0.10, 0.20, 0.25, 0.30, 0.35, 0.45, 0.50, 0.55, 0.60, 0.70])
    res = run_bernoulli_experiment(means=means, T=2000, n_seeds=20)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(1, 2001)
    for name in ["eps-greedy(0.1)", "UCB1", "Thompson", "Boltzmann(0.1)"]:
        shaded(ax, x, res[name], label=name)
    ax.set_xlabel("step $t$")
    ax.set_ylabel("cumulative regret $R_t$")
    ax.set_title("10-armed Bernoulli — mean $\\pm$ std over 20 seeds")
    ax.legend(loc="upper left")
    path = os.path.join(FIG_DIR, "fig_bernoulli_bakeoff.pdf")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_contextual_bandit() -> str:
    seed_everything(0)
    env = SyntheticLeadBrokerBandit(K=8, d=6, seed=0)
    res = run_contextual_experiment(env=env, T=1500, n_seeds=12, alpha=1.0)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(1, 1501)
    for name in ["LinUCB", "eps-greedy(0.1)"]:
        shaded(ax, x, res[name], label=name)
    ax.set_xlabel("step $t$")
    ax.set_ylabel("cumulative regret $R_t$")
    ax.set_title("Synthetic lead-broker contextual bandit\n(8 brokers, $d{=}6$)")
    ax.legend(loc="upper left")
    path = os.path.join(FIG_DIR, "fig_contextual_bandit.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_regret_bound() -> str:
    """Show empirical UCB1 regret vs the O((K log T)/Delta) upper bound."""
    seed_everything(0)
    means = np.array([0.5, 0.5 - 0.1, 0.5 - 0.2, 0.5 - 0.3])
    Ts = np.array([100, 200, 500, 1000, 2000, 5000])
    n_seeds = 10
    empirical = np.zeros((len(Ts), n_seeds))
    for i, T in enumerate(Ts):
        for s in range(n_seeds):
            from .env import BernoulliBandit
            from .agent import UCB1Agent
            env = BernoulliBandit(means=means, rng=np.random.default_rng(2000 + s))
            ag = UCB1Agent(K=4, c=2.0, seed=s)
            cum = 0.0
            for t in range(int(T)):
                a = ag.select(t + 1)
                r = env.pull(a)
                ag.update(a, r)
                cum += env.regret(a)
            empirical[i, s] = cum
    # bound: sum_a 8 log T / Delta_a + (pi^2/3) sum_a Delta_a (Auer et al. Theorem 1)
    deltas = means.max() - means
    suboptimal = deltas[deltas > 0]
    bound = np.array([
        8.0 * np.log(T) * np.sum(1.0 / suboptimal) + (1.0 + np.pi ** 2 / 3.0) * suboptimal.sum()
        for T in Ts
    ])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(Ts, empirical.mean(1), "o-", color=RL_PALETTE["blue"], label="empirical UCB1")
    ax.fill_between(
        Ts,
        empirical.mean(1) - empirical.std(1),
        empirical.mean(1) + empirical.std(1),
        color=RL_PALETTE["blue"], alpha=0.15,
    )
    ax.plot(Ts, bound, "--", color=RL_PALETTE["orange"],
            label="Auer et al. upper bound")
    ax.set_xscale("log")
    ax.set_xlabel("horizon $T$ (log)")
    ax.set_ylabel("cumulative regret $R_T$")
    ax.set_title("UCB1: empirical vs theoretical bound")
    ax.legend(loc="upper left")
    path = os.path.join(FIG_DIR, "fig_regret_bound.pdf")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    use_rl_style()
    paths = [fig_bernoulli_bakeoff(), fig_contextual_bandit(), fig_regret_bound()]
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
