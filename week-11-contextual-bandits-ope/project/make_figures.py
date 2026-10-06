"""Generate figures consumed by notes.tex.

Outputs (relative to week-11-contextual-bandits-ope/figs/):
    fig_regret.pdf       - cumulative regret of LinUCB and LinTS
    fig_ope_bias.pdf     - SNIPS / DR / Direct vs. ground truth
    fig_w_distribution.pdf - importance-weight histogram
    fig_bias_variance.pdf  - bias vs. variance trade-off (IPS vs DR vs SWITCH)

Deterministic given seed 0. Run as a script from the repo root:
    python week-11-contextual-bandits-ope/project/make_figures.py
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

# Allow `import project` style from within the week.
import project as P  # noqa: E402
from project.env import SyntheticLeadBandit  # noqa: E402
from project.agents import LinUCB, LinTS, EpsilonGreedyLogger  # noqa: E402
from project.ope import (  # noqa: E402
    RidgeRewardModel,
    ips,
    snips,
    direct_method,
    doubly_robust,
    switch_dr,
    true_policy_value,
)
from project.train import (  # noqa: E402
    run_online,
    collect_logs,
    softmax_policy,
    uniform_policy,
    greedy_oracle_policy,
)

FIGS = WEEK / "figs"
FIGS.mkdir(exist_ok=True)


def fig_regret(seed: int = 0, T: int = 2000) -> Path:
    use_rl_style()
    seed_everything(seed)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, build in (
        ("LinUCB ($\\alpha=1$)", lambda env: LinUCB(K=env.K, d=env.d, alpha=1.0)),
        ("LinTS ($v=0.25$)", lambda env: LinTS(K=env.K, d=env.d, v=0.25, seed=seed + 7)),
    ):
        seeds = [0, 1, 2, 3]
        curves = []
        for s in seeds:
            env = SyntheticLeadBandit(seed=seed + s)
            agent = build(env)
            _, reg = run_online(env, agent, T)
            curves.append(reg.cumsum())
        curves = np.stack(curves)
        m = curves.mean(0)
        lo = curves.mean(0) - curves.std(0)
        hi = curves.mean(0) + curves.std(0)
        x = np.arange(1, T + 1)
        line, = ax.plot(x, m, label=name)
        ax.fill_between(x, lo, hi, alpha=0.2, color=line.get_color(), linewidth=0)
    # Reference line: sqrt(T) growth and linear regret.
    ax.plot(x, 0.5 * np.sqrt(x * np.log(x + 1)), "--", color=RL_PALETTE["gray"],
            label=r"$\propto \sqrt{T \log T}$")
    ax.set_xlabel("round $t$")
    ax.set_ylabel("cumulative regret")
    ax.set_title("Synthetic lead-broker bandit\n(K=6, d=8): online regret")
    ax.legend(loc="upper left")
    out = FIGS / "fig_regret.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_ope_bias(seed: int = 0, n_log: int = 4000) -> Path:
    """Compare OPE estimators across three target policies, repeated runs."""
    use_rl_style()
    seed_everything(seed)

    R = 12
    ests = {"IPS": [], "SNIPS": [], "DR": [], "SWITCH-DR": [], "Direct": []}
    truths = []
    for s in range(R):
        env = SyntheticLeadBandit(seed=seed + s)
        log = collect_logs(env, n_log, 0.3, seed + 100 + s)
        qhat = RidgeRewardModel(K=env.K, d=env.d).fit(log)
        pi = softmax_policy(env, beta=4.0)
        truths.append(true_policy_value(env, pi, n=3000, seed=seed + 200 + s))
        ests["IPS"].append(ips(log, pi))
        ests["SNIPS"].append(snips(log, pi))
        ests["DR"].append(doubly_robust(log, pi, qhat))
        ests["SWITCH-DR"].append(switch_dr(log, pi, qhat, tau=10.0))
        ests["Direct"].append(direct_method(log, pi, qhat))
    truth = float(np.mean(truths))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    labels = list(ests.keys())
    means = [np.mean(ests[k]) for k in labels]
    errs = [np.std(ests[k]) for k in labels]
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=errs, capsize=4,
           color=[RL_PALETTE["blue"], RL_PALETTE["teal"], RL_PALETTE["orange"],
                  RL_PALETTE["gold"], RL_PALETTE["red"]])
    ax.axhline(truth, color=RL_PALETTE["gray"], linestyle="--",
               label=f"on-policy truth = {truth:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("estimated $V(\\pi_{\\mathrm{softmax}})$")
    ax.set_title("OPE estimators on synthetic insurance-bandit (mean $\\pm$ std over 12 seeds)")
    ax.legend()
    out = FIGS / "fig_ope_bias.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_w_distribution(seed: int = 0, n_log: int = 4000) -> Path:
    use_rl_style()
    seed_everything(seed)
    env = SyntheticLeadBandit(seed=seed)
    log = collect_logs(env, n_log, 0.3, seed + 1)
    pi_greedy = greedy_oracle_policy(env)
    pi_soft = softmax_policy(env, beta=4.0)
    w_g = np.array([pi_greedy(x)[a] / max(pb, 1e-12) for (x, a, _r, pb) in log])
    w_s = np.array([pi_soft(x)[a] / max(pb, 1e-12) for (x, a, _r, pb) in log])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bins = np.linspace(0, max(w_g.max(), w_s.max()) + 0.1, 50)
    ax.hist(w_s, bins=bins, alpha=0.65, label=r"$\pi=\mathrm{softmax}_{\beta=4}$",
            color=RL_PALETTE["teal"])
    ax.hist(w_g, bins=bins, alpha=0.65, label=r"$\pi=\mathrm{oracle\ greedy}$",
            color=RL_PALETTE["orange"])
    ax.set_xlabel("importance weight $w = \\pi(a|x)/\\pi_b(a|x)$")
    ax.set_ylabel("count")
    ax.set_title("Importance weight distribution under $\\varepsilon$-greedy logger")
    ax.legend()
    out = FIGS / "fig_w_distribution.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_bias_variance(seed: int = 0, n_log: int = 2000) -> Path:
    use_rl_style()
    seed_everything(seed)
    taus = [1.0, 2.0, 5.0, 10.0, 25.0, 100.0, 1000.0]
    R = 20
    means_dr = {tau: [] for tau in taus}
    truths = []
    for s in range(R):
        env = SyntheticLeadBandit(seed=seed + s)
        log = collect_logs(env, n_log, 0.3, seed + 100 + s)
        qhat = RidgeRewardModel(K=env.K, d=env.d).fit(log)
        pi = greedy_oracle_policy(env)
        truths.append(true_policy_value(env, pi, n=2000, seed=seed + 200 + s))
        for tau in taus:
            means_dr[tau].append(switch_dr(log, pi, qhat, tau=tau))
    truth = float(np.mean(truths))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bias = [abs(np.mean(means_dr[t]) - truth) for t in taus]
    var = [np.std(means_dr[t]) for t in taus]
    ax.plot(taus, bias, "-o", label="|bias|", color=RL_PALETTE["blue"])
    ax.plot(taus, var, "-s", label="std (variance proxy)", color=RL_PALETTE["orange"])
    ax.set_xscale("log")
    ax.set_xlabel(r"SWITCH threshold $\tau$")
    ax.set_ylabel("estimator error")
    ax.set_title("Bias-variance trade-off in SWITCH-DR (oracle target)")
    ax.legend()
    out = FIGS / "fig_bias_variance.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    print("[fig 1]", fig_regret())
    print("[fig 2]", fig_ope_bias())
    print("[fig 3]", fig_w_distribution())
    print("[fig 4]", fig_bias_variance())


if __name__ == "__main__":
    main()
