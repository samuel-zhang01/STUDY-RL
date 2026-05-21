"""Reference solutions for week-4 workbook exercises."""
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np

from .agents import (sarsa, q_learning, expected_sarsa, double_q_learning,
                     eps_greedy, make_eps_schedule, n_step_td_prediction)


# -------- Exercise 1: TD(0) for prediction --------------------------------

def exercise_1_td0(traj: List[Tuple[int, float, int]], V: np.ndarray,
                   alpha: float = 0.1, gamma: float = 1.0) -> np.ndarray:
    """Apply TD(0) updates V[s] += alpha (r + gamma V[s'] - V[s]) along a
    list of (s, r, s') tuples (terminal flagged by s' == -1).
    """
    V = V.copy()
    for s, r, sp in traj:
        target = r + (0.0 if sp == -1 else gamma * V[sp])
        V[s] += alpha * (target - V[s])
    return V


# -------- Exercise 2: bias-variance of n-step returns ---------------------

def exercise_2_nstep_return(rewards: List[float], V_terminal: float,
                            gamma: float, n: int) -> float:
    """Compute the n-step truncated return
        G_t^{(n)} = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n V(s_{t+n})
    given the next n rewards and the bootstrapped value of s_{t+n}.
    If len(rewards) < n, treat the trajectory as terminating early
    (V_terminal is ignored beyond episode end).
    """
    G = 0.0
    for k, r in enumerate(rewards[:n]):
        G += (gamma ** k) * r
    if len(rewards) >= n:
        G += (gamma ** n) * V_terminal
    return float(G)


# -------- Exercise 3: SARSA update single step ----------------------------

def exercise_3_sarsa_step(Q: np.ndarray, s: int, a: int, r: float,
                          sp: int, ap: int, done: bool, alpha: float, gamma: float) -> np.ndarray:
    Q = Q.copy()
    target = r + (0.0 if done else gamma * Q[sp, ap])
    Q[s, a] += alpha * (target - Q[s, a])
    return Q


# -------- Exercise 4: Q-learning update single step -----------------------

def exercise_4_qlearn_step(Q: np.ndarray, s: int, a: int, r: float,
                           sp: int, done: bool, alpha: float, gamma: float) -> np.ndarray:
    Q = Q.copy()
    target = r + (0.0 if done else gamma * Q[sp].max())
    Q[s, a] += alpha * (target - Q[s, a])
    return Q


# -------- Exercise 5: Expected-SARSA target -------------------------------

def exercise_5_expected_sarsa_target(Q_sp: np.ndarray, eps: float) -> float:
    """Compute the Expected-SARSA bootstrap value at next state s' under
    an eps-greedy policy w.r.t. Q_sp. Q_sp is shape (nA,).
    """
    nA = Q_sp.shape[0]
    best = int(np.argmax(Q_sp))
    probs = np.full(nA, eps / nA)
    probs[best] += 1 - eps
    return float(np.dot(probs, Q_sp))


# -------- Exercise 6: Double Q-learning step ------------------------------

def exercise_6_double_q_step(Q1: np.ndarray, Q2: np.ndarray, s: int, a: int,
                             r: float, sp: int, done: bool, alpha: float,
                             gamma: float, update_first: bool) -> Tuple[np.ndarray, np.ndarray]:
    Q1 = Q1.copy(); Q2 = Q2.copy()
    if update_first:
        a_star = int(np.argmax(Q1[sp]))
        target = r + (0.0 if done else gamma * Q2[sp, a_star])
        Q1[s, a] += alpha * (target - Q1[s, a])
    else:
        a_star = int(np.argmax(Q2[sp]))
        target = r + (0.0 if done else gamma * Q1[sp, a_star])
        Q2[s, a] += alpha * (target - Q2[s, a])
    return Q1, Q2


# -------- Exercise 7: TD-error variance vs MC error -----------------------

def exercise_7_td_error_var(rewards_per_episode: List[List[float]],
                            V_true: float, gamma: float = 1.0) -> Tuple[float, float]:
    """Empirical variance of TD(0) one-step targets vs Monte-Carlo returns
    for a *constant* policy whose state-value V_true(s) is known and
    state-stationary. Inputs are lists of per-episode reward sequences.
    Returns (var_TD, var_MC).
    """
    td_targets, mc_returns = [], []
    for rs in rewards_per_episode:
        # MC return for the *start* state
        G = 0.0
        for k, r in enumerate(rs):
            G += (gamma ** k) * r
        mc_returns.append(G)
        # one-step TD target from the start state
        td_targets.append(rs[0] + gamma * V_true)
    return float(np.var(td_targets, ddof=0)), float(np.var(mc_returns, ddof=0))


# -------- Exercise 8: cliff-walking bake-off ------------------------------

def exercise_8_bakeoff(make_env: Callable, n_episodes: int = 300, n_seeds: int = 3,
                       alpha: float = 0.5, eps_decay: int = 150) -> dict:
    out = {}
    for name, algo in [("sarsa", sarsa), ("q_learning", q_learning),
                       ("expected_sarsa", expected_sarsa)]:
        all_rs = []
        for s in range(n_seeds):
            env = make_env()
            _, rs = algo(env, n_episodes=n_episodes, alpha=alpha,
                         eps_decay=eps_decay, seed=s)
            all_rs.append(rs)
        out[name] = np.asarray(all_rs)
    return out


# -------- Exercise 9 (★★★): SNIPS off-policy estimator -------------------

def exercise_9_snips(rewards: np.ndarray, behaviour_probs: np.ndarray,
                     target_probs: np.ndarray) -> float:
    """Self-normalised importance-weighted estimator of E_target[R],
    given trajectories logged under a behaviour policy.

    All inputs are 1-D arrays of length N (one per trajectory / lead).
    """
    w = target_probs / np.maximum(behaviour_probs, 1e-12)
    return float((w * rewards).sum() / w.sum())
