"""Tabular TD agents specialised for week 4 mini-project.

These wrap a gymnasium discrete-state, discrete-action env. We keep the
implementations *vanilla* and *clear* — they're the reference for the
workbook exercises and for the empirical results figure.
"""
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np


# ---------- helpers -------------------------------------------------------

def make_eps_schedule(eps_start: float, eps_end: float, decay_episodes: int) -> Callable[[int], float]:
    """Linear epsilon decay over `decay_episodes`, then held at `eps_end`."""
    def sched(ep: int) -> float:
        if ep >= decay_episodes:
            return eps_end
        return eps_start + (eps_end - eps_start) * (ep / decay_episodes)
    return sched


def eps_greedy(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, Q.shape[1]))
    return int(np.argmax(Q[s]))


# ---------- algorithms (gymnasium-flavoured tabular) ----------------------

def _rollout_init(env, rng):
    seed = int(rng.integers(0, 2**31 - 1))
    out = env.reset(seed=seed)
    s = out[0] if isinstance(out, tuple) else out
    return int(s)


def sarsa(env, n_episodes: int = 500, alpha: float = 0.5, gamma: float = 1.0,
          eps_start: float = 1.0, eps_end: float = 0.1, eps_decay: int = 200,
          max_steps: int = 200, seed: int = 0) -> Tuple[np.ndarray, List[float]]:
    """On-policy TD control. Returns (Q, episode_returns)."""
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float64)
    sched = make_eps_schedule(eps_start, eps_end, eps_decay)
    returns: List[float] = []
    for ep in range(n_episodes):
        eps = sched(ep)
        s = _rollout_init(env, rng)
        a = eps_greedy(Q, s, eps, rng)
        G, done, t = 0.0, False, 0
        while not done and t < max_steps:
            sn, r, term, trunc, _ = env.step(a)
            sn = int(sn); done = bool(term or trunc)
            an = eps_greedy(Q, sn, eps, rng) if not term else 0
            target = r + (0.0 if term else gamma * Q[sn, an])
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = sn, an
            G += r; t += 1
        returns.append(G)
    return Q, returns


def q_learning(env, n_episodes: int = 500, alpha: float = 0.5, gamma: float = 1.0,
               eps_start: float = 1.0, eps_end: float = 0.1, eps_decay: int = 200,
               max_steps: int = 200, seed: int = 0) -> Tuple[np.ndarray, List[float]]:
    """Off-policy TD control. Returns (Q, episode_returns)."""
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float64)
    sched = make_eps_schedule(eps_start, eps_end, eps_decay)
    returns: List[float] = []
    for ep in range(n_episodes):
        eps = sched(ep)
        s = _rollout_init(env, rng)
        G, done, t = 0.0, False, 0
        while not done and t < max_steps:
            a = eps_greedy(Q, s, eps, rng)
            sn, r, term, trunc, _ = env.step(a)
            sn = int(sn); done = bool(term or trunc)
            target = r + (0.0 if term else gamma * Q[sn].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s = sn
            G += r; t += 1
        returns.append(G)
    return Q, returns


def expected_sarsa(env, n_episodes: int = 500, alpha: float = 0.5, gamma: float = 1.0,
                   eps_start: float = 1.0, eps_end: float = 0.1, eps_decay: int = 200,
                   max_steps: int = 200, seed: int = 0
                   ) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float64)
    sched = make_eps_schedule(eps_start, eps_end, eps_decay)
    returns: List[float] = []
    for ep in range(n_episodes):
        eps = sched(ep)
        s = _rollout_init(env, rng)
        G, done, t = 0.0, False, 0
        while not done and t < max_steps:
            a = eps_greedy(Q, s, eps, rng)
            sn, r, term, trunc, _ = env.step(a)
            sn = int(sn); done = bool(term or trunc)
            if not term:
                best = int(np.argmax(Q[sn]))
                probs = np.full(nA, eps / nA)
                probs[best] += 1 - eps
                exp_q = float(np.dot(probs, Q[sn]))
            else:
                exp_q = 0.0
            target = r + gamma * exp_q
            Q[s, a] += alpha * (target - Q[s, a])
            s = sn
            G += r; t += 1
        returns.append(G)
    return Q, returns


def double_q_learning(env, n_episodes: int = 500, alpha: float = 0.5, gamma: float = 1.0,
                      eps_start: float = 1.0, eps_end: float = 0.1, eps_decay: int = 200,
                      max_steps: int = 200, seed: int = 0
                      ) -> Tuple[np.ndarray, List[float]]:
    """Double Q-learning (van Hasselt 2010). Returns (Q1+Q2)/2, returns."""
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q1 = np.zeros((nS, nA)); Q2 = np.zeros((nS, nA))
    sched = make_eps_schedule(eps_start, eps_end, eps_decay)
    returns: List[float] = []
    for ep in range(n_episodes):
        eps = sched(ep)
        s = _rollout_init(env, rng)
        G, done, t = 0.0, False, 0
        while not done and t < max_steps:
            Qs = Q1[s] + Q2[s]
            if rng.random() < eps:
                a = int(rng.integers(0, nA))
            else:
                a = int(np.argmax(Qs))
            sn, r, term, trunc, _ = env.step(a)
            sn = int(sn); done = bool(term or trunc)
            if rng.random() < 0.5:
                # update Q1 using Q2 to evaluate the Q1-greedy action
                a_star = int(np.argmax(Q1[sn]))
                target = r + (0.0 if term else gamma * Q2[sn, a_star])
                Q1[s, a] += alpha * (target - Q1[s, a])
            else:
                a_star = int(np.argmax(Q2[sn]))
                target = r + (0.0 if term else gamma * Q1[sn, a_star])
                Q2[s, a] += alpha * (target - Q2[s, a])
            s = sn
            G += r; t += 1
        returns.append(G)
    return (Q1 + Q2) / 2.0, returns


def n_step_td_prediction(env, policy: Callable[[int, np.random.Generator], int],
                         n: int = 4, n_episodes: int = 300, alpha: float = 0.4,
                         gamma: float = 1.0, max_steps: int = 200, seed: int = 0
                         ) -> np.ndarray:
    """n-step TD evaluation; bootstrap at truncation, never at termination.

    The max_steps collection cap is also a truncation. policy(s, rng) -> a.
    """
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    nS = env.observation_space.n
    V = np.zeros(nS, dtype=np.float64)
    for _ in range(n_episodes):
        s = _rollout_init(env, rng)
        S = [s]; R = [0.0]  # R[0] unused; R[t] is reward after action t-1
        T = float("inf"); t = 0
        terminated = False
        while True:
            if t < T:
                a = policy(S[t], rng)
                sn, r, term, trunc, _ = env.step(a)
                S.append(int(sn)); R.append(float(r))
                if term or trunc or t + 1 >= max_steps:
                    T = t + 1
                    terminated = bool(term)
            tau = t - n + 1
            if tau >= 0:
                G = sum((gamma ** (i - tau - 1)) * R[i] for i in range(tau + 1, min(tau + n, T) + 1))
                if tau + n < T:
                    G += (gamma ** n) * V[S[tau + n]]
                elif not terminated:
                    G += (gamma ** (T - tau)) * V[S[int(T)]]
                V[S[tau]] += alpha * (G - V[S[tau]])
            if tau == T - 1:
                break
            t += 1
    return V
