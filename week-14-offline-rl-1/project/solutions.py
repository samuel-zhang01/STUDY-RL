"""Reference solutions for the workbook exercises.

Each function corresponds to one workbook exercise; the workbook imports
these in its hidden test cells.
"""
from __future__ import annotations
import numpy as np

from .agent import expectile_loss_grad, IQLAgent, CQLAgent, NaiveQAgent
from .env import collect_logged_dataset, make_env
from .train import train, evaluate_policy


def exercise_1_expectile(u: np.ndarray, tau: float) -> np.ndarray:
    """Return the per-sample expectile loss value (not gradient)."""
    w = np.where(u < 0, 1.0 - tau, tau)
    return w * u ** 2


def exercise_2_awr_weights(adv: np.ndarray, beta: float, clip: float = 100.0) -> np.ndarray:
    """Return normalised AWR sample weights: w_i ~ exp(beta * adv_i), capped."""
    # ``clip`` caps weights (not logits). Shift before exponentiating so large
    # positive advantages do not overflow; preserve the cap's relative scale.
    if clip <= 0 or not np.isfinite(clip):
        raise ValueError("weight cap must be finite and positive")
    logits = np.minimum(beta * np.asarray(adv, dtype=float), np.log(clip))
    w = np.exp(logits - logits.max())
    return w / w.mean()


def exercise_3_cql_penalty(Q_s: np.ndarray, a_beta: int) -> float:
    """log sum exp Q(s,.) - Q(s, a_beta), the discrete-action CQL penalty per state."""
    m = Q_s.max()
    lse = m + np.log(np.exp(Q_s - m).sum())
    return float(lse - Q_s[a_beta])


def exercise_4_collect_dataset(n_episodes: int = 50, seed: int = 0):
    return collect_logged_dataset(n_episodes=n_episodes, seed=seed)


def exercise_5_train_iql(seed: int = 0, n_steps: int = 1500):
    return train(seed=seed, algo="iql", n_steps=n_steps)


def exercise_6_train_cql(seed: int = 0, n_steps: int = 1500, alpha: float = 1.0):
    return train(seed=seed, algo="cql", n_steps=n_steps, cql_alpha=alpha)


def exercise_7_off_policy_evaluation_is(data: dict, target_actions: np.ndarray,
                                        gamma: float = 0.95) -> float:
    """Self-normalised importance-sampling (SNIPS) OPE.

    target_actions : np.ndarray of shape (nS,) giving greedy target-policy action per state.
    Returns trajectory SNIPS of the finite-horizon discounted return. Each
    episode's weight is the PRODUCT of per-step ratios. The target uses 1e-3
    smoothing; episode_end includes collection time limits. Long trajectories
    can still give low effective sample size despite stable log arithmetic.
    """
    done = np.asarray(data.get("episode_end", data["done"]), dtype=bool).copy()
    s_idx = data["s_idx"]; a = data["a"]; r = data["r"]; log_b = data["a_log_prob"]
    eps_target = 1e-3  # near-deterministic target
    nA = data["nA"]
    if not len(a):
        return float("nan")
    done[-1] = True  # include a final partial trajectory in legacy fixtures
    log_weights, returns = [], []
    log_w, G, gpow = 0.0, 0.0, 1.0
    for t in range(len(a)):
        # target prob: 1 - eps_target if matches, eps_target/(nA-1) otherwise (deterministic + smoothing).
        tgt_a = int(target_actions[s_idx[t]])
        if nA == 1:
            log_t = 0.0
        elif a[t] == tgt_a:
            log_t = np.log(1.0 - eps_target)
        else:
            log_t = np.log(eps_target / (nA - 1))
        log_w += log_t - log_b[t]
        G += gpow * r[t]
        gpow *= gamma
        if done[t]:
            log_weights.append(log_w); returns.append(G)
            log_w, G, gpow = 0.0, 0.0, 1.0
    log_weights = np.asarray(log_weights); returns = np.asarray(returns)
    if not np.isfinite(log_weights).any():
        return float("nan")
    weights = np.exp(log_weights - log_weights.max())
    return float((weights * returns).sum() / weights.sum())


def exercise_8_compare_algorithms(seed: int = 0, n_steps: int = 1500) -> dict:
    """Run IQL, CQL, and naive Q on the same dataset; return final returns."""
    out = {}
    for algo in ("naive", "cql", "iql"):
        out[algo] = train(seed=seed, algo=algo, n_steps=n_steps)["final_return"]
    return out
