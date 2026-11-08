"""Generate figures consumed by notes.tex.

Outputs (relative to week-11-contextual-bandits-ope/figs/):
    fig_regret.pdf       - cumulative regret of LinUCB and LinTS
    fig_ope_bias.pdf/.json - fixed-target OPE estimates and computation evidence
    fig_w_distribution.pdf - importance-weight histogram
    fig_bias_variance.pdf/.json - finite empirical squared-error decomposition

Deterministic given seed 0. Run as a script from the repo root:
    python week-11-contextual-bandits-ope/project/make_figures.py

Regenerate only the two fixed-environment statistical experiments, with their
documented defaults (12/20 independent logs and 50,000 reference contexts):
    python week-11-contextual-bandits-ope/project/make_figures.py --figure ope --figure switch
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
from hashlib import sha256
import json

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
ROOT = WEEK.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WEEK))

from common.plotting import use_rl_style, RL_PALETTE  # noqa: E402
from common.seed import seed_everything  # noqa: E402
from common.statistics import importance_weights  # noqa: E402

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
    independent_logged_sample,
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
    """Fixed softmax target, 12 independent adaptive logs and nuisance fits.

    The model seed fixes theta throughout. Error bars are sample SD across
    whole-log/fit replicates, not IID-row SE or a confidence interval. A shared
    50,000-context Monte Carlo reference separately records its sampling SE.
    """
    use_rl_style()
    seed_everything(seed)

    R = 12
    ests = {"IPS": [], "SNIPS": [], "DR": [], "SWITCH-DR": [], "Direct": []}
    fixed_env = fixed_reward_environment(seed, seed+1)
    pi = softmax_policy(fixed_env, beta=4.0)
    reference = context_mc_reference(fixed_env, pi, n=50000, seed=seed+33000)
    replicates = independent_replicate_seeds(seed+11000, R)
    for replicate in replicates:
        env = fixed_reward_environment(seed, replicate["data_seed"])
        np.testing.assert_array_equal(env.theta, fixed_env.theta)
        log = collect_logs(env, n_log, 0.3, replicate["logger_seed"])
        fit_log = independent_logged_sample(fixed_env, max(100, n_log//2), replicate["fit_seed"])
        qhat = RidgeRewardModel(K=env.K, d=env.d).fit(fit_log)
        ests["IPS"].append(ips(log, pi))
        ests["SNIPS"].append(snips(log, pi))
        ests["DR"].append(doubly_robust(log, pi, qhat))
        ests["SWITCH-DR"].append(switch_dr(log, pi, qhat, tau=10.0))
        ests["Direct"].append(direct_method(log, pi, qhat))
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    labels = list(ests.keys())
    means = [np.mean(ests[k]) for k in labels]
    errs = [np.std(ests[k], ddof=1) for k in labels]
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=errs, capsize=4,
           color=[RL_PALETTE["blue"], RL_PALETTE["teal"], RL_PALETTE["orange"],
                  RL_PALETTE["gold"], RL_PALETTE["red"]])
    ax.axhline(reference["estimate"], color=RL_PALETTE["gray"], linestyle="--",
               label=f"Context MC reference {reference['estimate']:.4f}; SE {reference['standard_error']:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Estimated expected reward")
    ax.set_title("Fixed softmax target: 12 independent logged datasets\nBars: mean $\\pm$ sample SD; nuisance fitting repeated")
    ax.legend(fontsize=9, frameon=True, facecolor="white", framealpha=1,
              edgecolor="#D6DCE5", borderpad=.6)
    out = FIGS / "fig_ope_bias.pdf"
    fig.tight_layout()
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)
    evidence = experiment_evidence(seed, n_log, R, fixed_env, reference, replicates)
    evidence.update(target={"kind": "softmax of fixed true logits", "beta": 4.0},
                    estimators=ests,
                    summaries={name: {"mean": float(np.mean(values)),
                                      "sample_sd_ddof1": float(np.std(values, ddof=1))}
                               for name, values in ests.items()},
                    error_bars="sample SD across independent whole-log and nuisance-fit replicates; not SE/CI")
    write_evidence(out.with_suffix('.json'), evidence)
    return out


def fig_w_distribution(seed: int = 0, n_log: int = 4000) -> Path:
    use_rl_style()
    seed_everything(seed)
    env = SyntheticLeadBandit(seed=seed)
    log = collect_logs(env, n_log, 0.3, seed + 1)
    pi_greedy = greedy_oracle_policy(env)
    pi_soft = softmax_policy(env, beta=4.0)
    logged_probabilities = np.array([pb for _x, _a, _r, pb in log])
    w_g = importance_weights([pi_greedy(x)[a] for x, a, _r, _pb in log], logged_probabilities)
    w_s = importance_weights([pi_soft(x)[a] for x, a, _r, _pb in log], logged_probabilities)

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
    estimates = {tau: [] for tau in taus}
    fixed_env = fixed_reward_environment(seed, seed+1)
    pi = greedy_oracle_policy(fixed_env)
    reference = context_mc_reference(fixed_env, pi, n=50000, seed=seed+33000)
    replicates = independent_replicate_seeds(seed+22000, R)
    for replicate in replicates:
        env = fixed_reward_environment(seed, replicate["data_seed"])
        np.testing.assert_array_equal(env.theta, fixed_env.theta)
        log = collect_logs(env, n_log, 0.3, replicate["logger_seed"])
        fit_log = independent_logged_sample(fixed_env, max(100, n_log//2), replicate["fit_seed"])
        qhat = RidgeRewardModel(K=env.K, d=env.d).fit(fit_log)
        for tau in taus:
            estimates[tau].append(switch_dr(log, pi, qhat, tau=tau))
    summaries = {tau: empirical_error_moments(values, reference["estimate"])
                 for tau, values in estimates.items()}
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(taus, [summaries[t]["squared_mean_error"] for t in taus], "-o",
            label="Squared empirical mean error", color=RL_PALETTE["blue"])
    ax.plot(taus, [summaries[t]["variance_ddof0"] for t in taus], "-s",
            label="Empirical variance (divisor 20)", color=RL_PALETTE["orange"])
    ax.plot(taus, [summaries[t]["mse"] for t in taus], "--^",
            label="Empirical MSE", color=RL_PALETTE["teal"])
    ax.set_xscale("log")
    ax.set_xlabel(r"SWITCH threshold $\tau$")
    ax.set_ylabel("Squared expected-reward units")
    ax.set_title("Fixed oracle target: 20 independent logged datasets\nFinite empirical decomposition relative to context MC reference")
    ax.legend(fontsize=9)
    out = FIGS / "fig_bias_variance.pdf"
    fig.tight_layout()
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)
    evidence = experiment_evidence(seed, n_log, R, fixed_env, reference, replicates)
    evidence.update(target={"kind": "greedy action under fixed true reward mechanism"},
                    thresholds=taus,
                    estimates={str(tau): values for tau, values in estimates.items()},
                    summaries={str(tau): summary for tau, summary in summaries.items()},
                    squared_units="squared expected-reward units",
                    identity="MSE = squared empirical mean error + variance with divisor R",
                    interpretation="finite empirical moments relative to one shared MC approximation; not population bias/variance or a universal ordering")
    write_evidence(out.with_suffix('.json'), evidence)
    return out


def fixed_reward_environment(model_seed: int, data_seed: int) -> SyntheticLeadBandit:
    """Keep the reward mechanism fixed while independently drawing a new log.

    SyntheticLeadBandit.seed normally controls theta AND data. Resetting only
    its observation generator avoids confounding environment heterogeneity
    with estimator sampling variance. The target is created once outside the
    replicate loop from the same fixed theta.
    """
    env = SyntheticLeadBandit(seed=model_seed)
    env._rng = np.random.default_rng(data_seed)
    return env


def independent_replicate_seeds(seed: int, repetitions: int) -> list[dict]:
    """Separate SeedSequence child streams for data, adaptive logger and fit."""
    streams = np.random.SeedSequence(seed).spawn(3*repetitions)
    return [{"replicate": index,
             "data_seed": int(streams[3*index].generate_state(1)[0]),
             "logger_seed": int(streams[3*index+1].generate_state(1)[0]),
             "fit_seed": int(streams[3*index+2].generate_state(1)[0])}
            for index in range(repetitions)]


def context_mc_reference(env, pi, n: int = 50000, seed: int = 33000) -> dict:
    """Approximate E_X[sum_a pi(a|X)q(X,a)] using independent fresh contexts.

    Conditional reward means integrate out action and Bernoulli-outcome noise.
    Context integration is still Monte Carlo: report its sample SD/sqrt(n).
    This function does not consume any evaluation or fitting RNG state.
    """
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("reference needs at least two independent contexts")
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    for index in range(n):
        x = rng.normal(size=env.d)
        x[-1] = 1.
        x /= max(1., np.linalg.norm(x))
        values[index] = float(pi(x) @ env.all_means(x))
    if not np.isfinite(values).all():
        raise ValueError("reference conditional means must be finite")
    sd = float(values.std(ddof=1))
    return {"estimate": float(values.mean()), "standard_error": sd/np.sqrt(n),
            "sample_sd": sd, "n_contexts": int(n), "seed": int(seed),
            "quantity": "E_X[sum_a pi(a|X) q(X,a)]",
            "method": "fresh IID context Monte Carlo; actions and rewards integrated via simulator conditional means",
            "units": "expected Bernoulli reward", "exact_truth": False,
            "conditional_values_sha256": sha256(values.tobytes()).hexdigest()}


def empirical_error_moments(estimates, reference: float) -> dict:
    """Finite empirical squared-error identity around a shared reference.

    The variance uses divisor R so that the identity holds for the recorded
    repetitions. Squared empirical mean error is not a claim of known squared
    population bias, and the Monte Carlo reference has its own uncertainty.
    """
    values = np.asarray(estimates, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or not np.isfinite(reference):
        raise ValueError("need at least two finite estimates and a finite shared reference")
    errors = values-reference
    mean_error = float(errors.mean())
    squared_mean_error = mean_error**2
    variance = float(np.mean((errors-mean_error)**2))
    mse = float(np.mean(errors**2))
    if not np.isclose(mse, squared_mean_error+variance, rtol=1e-12, atol=1e-15):
        raise AssertionError("finite empirical MSE decomposition failed")
    return {"n_repetitions": len(values), "mean_estimate": float(values.mean()),
            "mean_error": mean_error, "squared_mean_error": squared_mean_error,
            "variance_ddof0": variance, "mse": mse,
            "identity_residual": mse-squared_mean_error-variance}


def experiment_evidence(model_seed, n_log, repetitions, env, reference, replicates):
    return {"schema": 1, "producer": "project/make_figures.py",
            "producer_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_seed": int(model_seed), "theta_sha256": sha256(env.theta.tobytes()).hexdigest(),
            "repetitions": repetitions, "evaluation_rows_per_log": n_log,
            "fitting_rows_per_replicate": max(100,n_log//2),
            "independent_unit": "a complete adaptive logged dataset and its independently generated nuisance-fitting dataset",
            "within_log": "adaptive epsilon-greedy actions; rows are not asserted IID",
            "logger_epsilon": .3, "reward_model": "per-arm ridge, lambda=1",
            "reference": reference, "replicate_seeds": replicates,
            "units": "expected Bernoulli reward"}


def write_evidence(path, evidence):
    path.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', action='append', choices=['regret','ope','weights','switch'])
    args = parser.parse_args()
    selected = args.figure or ['regret','ope','weights','switch']
    producers = dict(regret=fig_regret,ope=fig_ope_bias,weights=fig_w_distribution,switch=fig_bias_variance)
    for name in selected:
        print(name, producers[name](), flush=True)


if __name__ == "__main__":
    main()
