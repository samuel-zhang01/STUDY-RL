"""Offline dataset construction.

We generate a tabular logged dataset on a 5x5 GridWorld using a mixture
behaviour policy: an epsilon-greedy logger over the optimal policy plus
a pure-random subset. This produces *narrow but suboptimal* coverage,
which is the canonical setting where naive Q-learning suffers from
extrapolation error.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.gridworld import GridWorld
from common.tabular import value_iteration


def make_env(seed: int = 0) -> GridWorld:
    """A 5x5 GridWorld with a pit at (2,2) and goal at (4,4)."""
    return GridWorld(rows=5, cols=5, start=(0, 0), goals=((4, 4),), pits=((2, 2),),
                     slip_prob=0.1, rng=np.random.default_rng(seed))


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def _epsilon_greedy(Q: np.ndarray, s_idx: int, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, Q.shape[1]))
    return int(np.argmax(Q[s_idx]))


def collect_logged_dataset(
    n_episodes: int = 200,
    max_steps: int = 30,
    eps_logger: float = 0.3,
    random_frac: float = 0.2,
    seed: int = 0,
):
    """Collect a logged offline dataset with mixture behaviour policy.

    Returns dict of numpy arrays with keys s, a, r, s2, done, s_idx, s2_idx, a_log_prob.
    States are one-hot (nS=25). Actions are discrete in {0,1,2,3}.
    """
    env = make_env(seed=seed)
    nS, nA = env.nS, env.nA
    # Pre-solve the MDP to extract a near-optimal policy for the logger.
    V_star, pi_star, _ = value_iteration(env, gamma=0.95, tol=1e-8)
    # Build a Q*-like table for action selection (one-hot greedy from pi_star).
    Q_star = np.zeros((nS, nA), dtype=np.float32)
    for s in range(nS):
        Q_star[s, int(pi_star[s])] = 1.0
    rng = np.random.default_rng(seed)

    S, A, R, S2, D, SI, SI2, LP = [], [], [], [], [], [], [], []
    for ep in range(n_episodes):
        use_random = rng.random() < random_frac
        eps = 1.0 if use_random else eps_logger
        env.reset(seed=seed + ep)
        s = env.state
        for t in range(max_steps):
            s_idx = env.s2idx(s)
            a = _epsilon_greedy(Q_star, s_idx, eps, rng)
            # Probability under the logger
            greedy_a = int(np.argmax(Q_star[s_idx]))
            prob = (1.0 - eps) * (1.0 if a == greedy_a else 0.0) + eps / nA
            s2, r, done, _ = env.step(a)
            s2_idx = env.s2idx(s2)
            S.append(_onehot(s_idx, nS)); A.append(a); R.append(r)
            S2.append(_onehot(s2_idx, nS)); D.append(float(done))
            SI.append(s_idx); SI2.append(s2_idx); LP.append(np.log(prob + 1e-12))
            s = s2
            if done:
                break

    return {
        "s": np.asarray(S, dtype=np.float32),
        "a": np.asarray(A, dtype=np.int64),
        "r": np.asarray(R, dtype=np.float32),
        "s2": np.asarray(S2, dtype=np.float32),
        "done": np.asarray(D, dtype=np.float32),
        "s_idx": np.asarray(SI, dtype=np.int64),
        "s2_idx": np.asarray(SI2, dtype=np.int64),
        "a_log_prob": np.asarray(LP, dtype=np.float32),
        "nS": nS,
        "nA": nA,
        "Q_star": Q_star,
    }
