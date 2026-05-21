"""Tabular RL primitives: value iteration, policy iteration, MC, SARSA, Q-learning,
Expected SARSA, Double Q-learning. Reused across the foundation lectures (weeks 2-4)."""
from __future__ import annotations
from collections import defaultdict
from typing import Callable, Dict, Tuple, List, Optional
import numpy as np

from .gridworld import GridWorld


# ===================== Dynamic Programming =====================

def value_iteration(env: GridWorld, gamma: float = 0.99, tol: float = 1e-8,
                    max_iter: int = 10_000) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Synchronous value iteration. Returns (V, greedy_policy, deltas)."""
    P = env.transition_table()
    V = np.zeros(env.nS)
    deltas: List[float] = []
    for _ in range(max_iter):
        V_new = np.zeros_like(V)
        for s in range(env.nS):
            qs = np.full(env.nA, -np.inf)
            for a in range(env.nA):
                qs[a] = sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[(s, a)])
            V_new[s] = qs.max()
        delta = np.max(np.abs(V_new - V))
        deltas.append(float(delta))
        V = V_new
        if delta < tol:
            break
    # extract greedy policy
    pi = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        qs = np.zeros(env.nA)
        for a in range(env.nA):
            qs[a] = sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[(s, a)])
        pi[s] = int(qs.argmax())
    return V, pi, deltas


def policy_evaluation(env: GridWorld, policy: np.ndarray, gamma: float = 0.99,
                      tol: float = 1e-8, max_iter: int = 10_000) -> np.ndarray:
    """Iterative policy evaluation for a deterministic policy."""
    P = env.transition_table()
    V = np.zeros(env.nS)
    for _ in range(max_iter):
        V_new = np.zeros_like(V)
        for s in range(env.nS):
            a = int(policy[s])
            V_new[s] = sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[(s, a)])
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V


def policy_iteration(env: GridWorld, gamma: float = 0.99,
                     max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
    """Howard's policy iteration. Returns (V, pi, n_sweeps)."""
    P = env.transition_table()
    pi = np.zeros(env.nS, dtype=int)
    for sweep in range(max_iter):
        V = policy_evaluation(env, pi, gamma=gamma)
        stable = True
        for s in range(env.nS):
            qs = np.zeros(env.nA)
            for a in range(env.nA):
                qs[a] = sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[(s, a)])
            best = int(qs.argmax())
            if best != pi[s]:
                stable = False
                pi[s] = best
        if stable:
            return V, pi, sweep + 1
    return V, pi, max_iter


# ===================== Model-free control =====================

def _eps_greedy(Q: Dict[Tuple[int, int], float], s_idx: int, nA: int, eps: float,
                rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, nA))
    qs = np.array([Q[(s_idx, a)] for a in range(nA)])
    return int(qs.argmax())


def q_learning(env: GridWorld, n_episodes: int = 5000, alpha: float = 0.1,
               gamma: float = 0.99, eps_start: float = 1.0, eps_end: float = 0.05,
               eps_decay_episodes: int = 2000, seed: int = 0
               ) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, List[float]]:
    """Tabular Q-learning (off-policy TD control)."""
    rng = np.random.default_rng(seed)
    Q: Dict[Tuple[int, int], float] = defaultdict(float)
    returns: List[float] = []
    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)
        s = env.reset(seed=int(rng.integers(0, 2**31)))
        s_idx = env.s2idx(s)
        done = False
        G = 0.0
        steps = 0
        while not done and steps < 500:
            a = _eps_greedy(Q, s_idx, env.nA, eps, rng)
            (ns, r, done, _) = env.step(a)
            ns_idx = env.s2idx(ns)
            target = r + (0.0 if done else gamma * max(Q[(ns_idx, ap)] for ap in range(env.nA)))
            Q[(s_idx, a)] += alpha * (target - Q[(s_idx, a)])
            s_idx = ns_idx
            G += r
            steps += 1
        returns.append(G)
    pi = np.array([int(np.argmax([Q[(s, a)] for a in range(env.nA)])) for s in range(env.nS)])
    return Q, pi, returns


def sarsa(env: GridWorld, n_episodes: int = 5000, alpha: float = 0.1, gamma: float = 0.99,
          eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_episodes: int = 2000,
          seed: int = 0) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, List[float]]:
    """Tabular SARSA (on-policy TD control)."""
    rng = np.random.default_rng(seed)
    Q: Dict[Tuple[int, int], float] = defaultdict(float)
    returns: List[float] = []
    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)
        s = env.reset(seed=int(rng.integers(0, 2**31)))
        s_idx = env.s2idx(s)
        a = _eps_greedy(Q, s_idx, env.nA, eps, rng)
        done = False
        G = 0.0
        steps = 0
        while not done and steps < 500:
            (ns, r, done, _) = env.step(a)
            ns_idx = env.s2idx(ns)
            ap = _eps_greedy(Q, ns_idx, env.nA, eps, rng) if not done else 0
            target = r + (0.0 if done else gamma * Q[(ns_idx, ap)])
            Q[(s_idx, a)] += alpha * (target - Q[(s_idx, a)])
            s_idx, a = ns_idx, ap
            G += r
            steps += 1
        returns.append(G)
    pi = np.array([int(np.argmax([Q[(s, a)] for a in range(env.nA)])) for s in range(env.nS)])
    return Q, pi, returns


def expected_sarsa(env: GridWorld, n_episodes: int = 5000, alpha: float = 0.1,
                   gamma: float = 0.99, eps_start: float = 1.0, eps_end: float = 0.05,
                   eps_decay_episodes: int = 2000, seed: int = 0
                   ) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    Q: Dict[Tuple[int, int], float] = defaultdict(float)
    returns: List[float] = []
    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)
        s = env.reset(seed=int(rng.integers(0, 2**31)))
        s_idx = env.s2idx(s)
        done = False
        G = 0.0
        steps = 0
        while not done and steps < 500:
            a = _eps_greedy(Q, s_idx, env.nA, eps, rng)
            (ns, r, done, _) = env.step(a)
            ns_idx = env.s2idx(ns)
            q_next = np.array([Q[(ns_idx, ap)] for ap in range(env.nA)])
            best = int(q_next.argmax())
            probs = np.full(env.nA, eps / env.nA)
            probs[best] += 1 - eps
            expected = float((probs * q_next).sum())
            target = r + (0.0 if done else gamma * expected)
            Q[(s_idx, a)] += alpha * (target - Q[(s_idx, a)])
            s_idx = ns_idx
            G += r
            steps += 1
        returns.append(G)
    pi = np.array([int(np.argmax([Q[(s, a)] for a in range(env.nA)])) for s in range(env.nS)])
    return Q, pi, returns


def mc_control_epsgreedy(env: GridWorld, n_episodes: int = 5000, gamma: float = 0.99,
                         eps_start: float = 1.0, eps_end: float = 0.05,
                         eps_decay_episodes: int = 2000, seed: int = 0
                         ) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, List[float]]:
    """Every-visit Monte Carlo control with epsilon-greedy."""
    rng = np.random.default_rng(seed)
    Q: Dict[Tuple[int, int], float] = defaultdict(float)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    returns: List[float] = []
    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)
        s = env.reset(seed=int(rng.integers(0, 2**31)))
        traj: List[Tuple[int, int, float]] = []
        done = False
        steps = 0
        while not done and steps < 500:
            s_idx = env.s2idx(s)
            a = _eps_greedy(Q, s_idx, env.nA, eps, rng)
            ns, r, done, _ = env.step(a)
            traj.append((s_idx, a, r))
            s = ns
            steps += 1
        G = 0.0
        for (s_idx, a, r) in reversed(traj):
            G = r + gamma * G
            counts[(s_idx, a)] += 1
            Q[(s_idx, a)] += (G - Q[(s_idx, a)]) / counts[(s_idx, a)]
        returns.append(sum(r for _, _, r in traj))
    pi = np.array([int(np.argmax([Q[(s, a)] for a in range(env.nA)])) for s in range(env.nS)])
    return Q, pi, returns
