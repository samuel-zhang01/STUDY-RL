"""Reference solutions for the week-03 workbook."""
from __future__ import annotations
from collections import defaultdict
from typing import Callable, Dict, List, Tuple
import numpy as np

from .blackjack import BlackjackEnv
from .recruitment import RecruitmentMDP


# ============================================================================
# Exercise 1 (★) — Returns from a trajectory (first-visit)
# ============================================================================
def exercise_1_first_visit_returns(
    trajectory: List[Tuple[int, int, float]], gamma: float
) -> Dict[Tuple[int, int], List[float]]:
    """Given a trajectory [(s,a,r), ...], return for each (s,a) the list of
    first-visit Monte-Carlo returns (each visit contributes the one return
    seen at the *first* occurrence inside this trajectory)."""
    seen = set()
    returns: Dict[Tuple[int, int], List[float]] = {}
    # Compute discounted returns from each timestep onwards in O(T).
    Gs = [0.0] * len(trajectory)
    G = 0.0
    for t in range(len(trajectory) - 1, -1, -1):
        G = trajectory[t][2] + gamma * G
        Gs[t] = G
    for t, (s, a, _r) in enumerate(trajectory):
        if (s, a) in seen:
            continue
        seen.add((s, a))
        returns.setdefault((s, a), []).append(Gs[t])
    return returns


# ============================================================================
# Exercise 2 (★) — Every-visit MC prediction on a fixed policy
# ============================================================================
def exercise_2_every_visit_mc(env, policy: Callable[[Tuple[int, int]], int],
                              n_episodes: int = 5000, gamma: float = 1.0,
                              seed: int = 0) -> Dict[Tuple[int, int, int], float]:
    """Estimate V^pi by every-visit MC. Returns dict obs -> V estimate."""
    rng = np.random.default_rng(seed)
    sums: Dict = defaultdict(float)
    counts: Dict = defaultdict(int)
    for ep in range(n_episodes):
        obs = env.reset(seed=int(rng.integers(0, 2**31)))
        traj: List[Tuple[Tuple, float]] = []
        done = False
        steps = 0
        while not done and steps < 200:
            a = int(policy(obs))
            new_obs, r, done, _ = env.step(a)
            traj.append((obs, r))
            obs = new_obs
            steps += 1
        G = 0.0
        for obs_t, r in reversed(traj):
            G = r + gamma * G
            sums[obs_t] += G
            counts[obs_t] += 1
    return {k: sums[k] / counts[k] for k in sums}


# ============================================================================
# Exercise 3 (★★) — On-policy MC control with eps-soft policy on Blackjack
# ============================================================================
def exercise_3_blackjack_mc_control(n_episodes: int = 50_000, gamma: float = 1.0,
                                    eps: float = 0.1, seed: int = 0
                                    ) -> Tuple[Dict, Dict, List[float]]:
    """Every-visit MC control with epsilon-greedy on Blackjack. Returns
    (Q, policy, returns_per_ep)."""
    rng = np.random.default_rng(seed)
    env = BlackjackEnv(seed=seed)
    Q: Dict = defaultdict(float)
    counts: Dict = defaultdict(int)
    pi: Dict = {}
    returns: List[float] = []
    for ep in range(n_episodes):
        obs = env.reset(seed=int(rng.integers(0, 2**31)))
        traj: List[Tuple[Tuple, int, float]] = []
        done = False
        while not done:
            if rng.random() < eps or obs not in pi:
                a = int(rng.integers(0, 2))
            else:
                a = pi[obs]
            new_obs, r, done, _ = env.step(a)
            traj.append((obs, a, r))
            obs = new_obs
        G = 0.0
        for s, a, r in reversed(traj):
            G = r + gamma * G
            counts[(s, a)] += 1
            Q[(s, a)] += (G - Q[(s, a)]) / counts[(s, a)]
        # update greedy policy at visited states only
        for s, _a, _r in traj:
            qs = [Q[(s, 0)], Q[(s, 1)]]
            pi[s] = int(np.argmax(qs))
        returns.append(sum(r for _, _, r in traj))
    return dict(Q), pi, returns


# ============================================================================
# Exercise 4 (★★) — Ordinary importance sampling estimator
# ============================================================================
def exercise_4_ordinary_is(
    rewards: List[float], target_prob: List[float], logged_prob: List[float]
) -> float:
    """OIS: V_hat(pi) = mean( prod_t (pi/pi_b)_t * G_i ) for trajectories
    in flat form: each entry is (G, w) for one trajectory.

    The simpler one-step bandit-style version is used here for clarity:
    V = mean( pi/pi_b * r ).
    """
    r = np.asarray(rewards, dtype=float)
    w = np.asarray(target_prob, dtype=float) / np.maximum(np.asarray(logged_prob, dtype=float), 1e-12)
    return float((w * r).mean())


# ============================================================================
# Exercise 5 (★★) — Weighted (self-normalised) importance sampling
# ============================================================================
def exercise_5_weighted_is(
    rewards: List[float], target_prob: List[float], logged_prob: List[float]
) -> float:
    r = np.asarray(rewards, dtype=float)
    w = np.asarray(target_prob, dtype=float) / np.maximum(np.asarray(logged_prob, dtype=float), 1e-12)
    return float((w * r).sum() / max(w.sum(), 1e-12))


# ============================================================================
# Exercise 6 (★★) — IS variance vs WIS variance
# ============================================================================
def exercise_6_compare_is_wis_variance(
    n_trials: int = 200, n_logs: int = 200, seed: int = 0
) -> Tuple[float, float]:
    """Run OIS and WIS many times on a tiny bandit; return (var_OIS, var_WIS).
    WIS should have substantially lower variance."""
    rng = np.random.default_rng(seed)
    p_b = np.array([0.5, 0.5])  # behaviour
    p_t = np.array([0.9, 0.1])  # target
    true_means = np.array([0.7, 0.2])
    ois_vals = np.zeros(n_trials)
    wis_vals = np.zeros(n_trials)
    for trial in range(n_trials):
        rewards = []; tp = []; lp = []
        for _ in range(n_logs):
            a = int(rng.choice(2, p=p_b))
            r = float(rng.random() < true_means[a])
            rewards.append(r); tp.append(p_t[a]); lp.append(p_b[a])
        ois_vals[trial] = exercise_4_ordinary_is(rewards, tp, lp)
        wis_vals[trial] = exercise_5_weighted_is(rewards, tp, lp)
    return float(np.var(ois_vals)), float(np.var(wis_vals))


# ============================================================================
# Exercise 7 (★★★, open-ended) — Off-policy MC with per-decision IS on
# Recruitment MDP
# ============================================================================
def exercise_7_offpolicy_recruitment(n_episodes: int = 4000, seed: int = 0
                                     ) -> Tuple[float, float, float]:
    """Estimate the value of a hand-coded target policy ('always advance')
    using logs from a uniform-random behaviour policy. Returns
    (V_OIS, V_WIS, V_MC_oracle).
    """
    rng = np.random.default_rng(seed)
    env = RecruitmentMDP(seed=seed)
    K = env.nA
    # Behaviour: uniform over {reject, advance}.
    # Target: always advance.
    rewards, target_p, logged_p = [], [], []
    for _ in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31)))
        obs = env._obs()
        done = False
        cum_target_prob = 1.0
        cum_logged_prob = 1.0
        G = 0.0
        while not done:
            a = int(rng.integers(0, K))
            cum_logged_prob *= 1.0 / K
            cum_target_prob *= 1.0 if a == 1 else 0.0
            new_obs, r, done, _ = env.step(a)
            G += r
            obs = new_obs
        rewards.append(G); target_p.append(cum_target_prob); logged_p.append(cum_logged_prob)
    v_ois = exercise_4_ordinary_is(rewards, target_p, logged_p)
    v_wis = exercise_5_weighted_is(rewards, target_p, logged_p)
    # Oracle: directly Monte Carlo the target.
    n_mc = 4000; total = 0.0
    for _ in range(n_mc):
        env.reset(seed=int(rng.integers(0, 2**31)))
        done = False; G = 0.0
        while not done:
            _, r, done, _ = env.step(1)
            G += r
        total += G
    v_oracle = total / n_mc
    return float(v_ois), float(v_wis), float(v_oracle)


# ============================================================================
# Exercise 8 (★★) — Variance of the importance-sampled return
# ============================================================================
def exercise_8_is_variance_bound(p_target: np.ndarray, p_logged: np.ndarray,
                                 returns: np.ndarray) -> Tuple[float, float]:
    """Return (Var[w r] under behaviour, max-weight-squared * E[r^2] upper bound).
    Verifies the textbook bound Var(IS) <= w_max^2 * E[r^2]."""
    w = np.asarray(p_target) / np.maximum(np.asarray(p_logged), 1e-12)
    r = np.asarray(returns, dtype=float)
    return float(np.var(w * r)), float((w.max() ** 2) * (r ** 2).mean())
