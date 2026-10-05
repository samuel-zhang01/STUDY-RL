"""Reference solutions for the Week 2 workbook exercises."""
from __future__ import annotations
from typing import Tuple, List
import numpy as np

from common.gridworld import GridWorld
from common.tabular import value_iteration, policy_iteration, policy_evaluation
from .carrental import CarRentalMDP


# Ex 1 — apply the Bellman operator once
def exercise_1_bellman_operator(env: GridWorld, V: np.ndarray,
                                gamma: float = 0.99) -> np.ndarray:
    """Return (T_* V)(s) = max_a sum_s' p(s'|s,a)(r + gamma V(s'))."""
    P = env.transition_table()
    V_new = np.zeros_like(V)
    for s in range(env.nS):
        qs = []
        for a in range(env.nA):
            qs.append(sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[(s, a)]))
        V_new[s] = max(qs)
    return V_new


# Ex 2 — value iteration on a 5x5 deterministic GridWorld
def exercise_2_value_iteration(rows: int = 5, cols: int = 5,
                               gamma: float = 0.95) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    env = GridWorld(rows=rows, cols=cols, start=(0, 0),
                    goals=((rows - 1, cols - 1),))
    V, pi, deltas = value_iteration(env, gamma=gamma)
    return V, pi, deltas


# Ex 3 — policy iteration converges in few sweeps
def exercise_3_policy_iteration(rows: int = 5, cols: int = 5,
                                gamma: float = 0.95) -> Tuple[np.ndarray, np.ndarray, int]:
    env = GridWorld(rows=rows, cols=cols, start=(0, 0),
                    goals=((rows - 1, cols - 1),))
    V, pi, n = policy_iteration(env, gamma=gamma)
    return V, pi, n


# Ex 4 — stochastic GridWorld (slip)
def exercise_4_slip_world_optimal_value(slip: float = 0.2,
                                        gamma: float = 0.95) -> float:
    env = GridWorld(rows=4, cols=4, start=(0, 0),
                    goals=((3, 3),), pits=((1, 1),),
                    slip_prob=slip)
    V, _, _ = value_iteration(env, gamma=gamma)
    return float(V[env.s2idx((0, 0))])


# Ex 5 — Bellman residual & contraction
def exercise_5_contraction_factor(env: GridWorld, V1: np.ndarray, V2: np.ndarray,
                                  gamma: float = 0.95) -> Tuple[float, float]:
    """Return (||TV1 - TV2||_inf, ||V1 - V2||_inf). Their ratio must be <= gamma."""
    TV1 = exercise_1_bellman_operator(env, V1, gamma=gamma)
    TV2 = exercise_1_bellman_operator(env, V2, gamma=gamma)
    return float(np.max(np.abs(TV1 - TV2))), float(np.max(np.abs(V1 - V2)))


# Ex 6 — policy evaluation matches the closed-form linear system
def exercise_6_policy_evaluation_closed_form(env: GridWorld, policy: np.ndarray,
                                             gamma: float = 0.9) -> np.ndarray:
    """Solve V = (I - gamma P_pi)^-1 r_pi as a single dense linear system."""
    P = env.transition_table()
    nS = env.nS
    P_pi = np.zeros((nS, nS))
    r_pi = np.zeros(nS)
    for s in range(nS):
        a = int(policy[s])
        for p, ns, r, d in P[(s, a)]:
            P_pi[s, ns] += p * (0.0 if d else 1.0)
            r_pi[s] += p * r
    V = np.linalg.solve(np.eye(nS) - gamma * P_pi, r_pi)
    return V


# Ex 7 — Generalised policy iteration with truncated evaluation
def exercise_7_gpi(env: GridWorld, gamma: float = 0.95,
                   eval_sweeps: int = 1, max_outer: int = 50, tol: float = 1e-8
                   ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Modified policy iteration: truncated evaluation, then improve.

    Policy stability alone is insufficient with truncated evaluation; stop
    only when the optimal Bellman residual is also small. A max_outer exit
    returns the current approximation rather than a convergence certificate.
    """
    if eval_sweeps < 1 or max_outer < 1:
        raise ValueError("eval_sweeps and max_outer must be positive")
    P = env.transition_table()
    pi = np.zeros(env.nS, dtype=int)
    V = np.zeros(env.nS)
    for outer in range(max_outer):
        # truncated evaluation
        for _ in range(eval_sweeps):
            V_new = np.zeros_like(V)
            for s in range(env.nS):
                a = int(pi[s])
                V_new[s] = sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[(s, a)])
            V = V_new
        # improvement
        stable = True
        for s in range(env.nS):
            qs = np.zeros(env.nA)
            for a in range(env.nA):
                qs[a] = sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[(s, a)])
            best = int(qs.argmax())
            if best != pi[s]:
                stable = False
                pi[s] = best
        residual = np.max(np.abs(exercise_1_bellman_operator(env, V, gamma) - V))
        if stable and residual < tol:
            return V, pi, outer + 1
    return V, pi, max_outer


# Ex 8 — Jack's car rental, value iteration
def exercise_8_car_rental(cap: int = 5, gamma: float = 0.9
                          ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run value iteration on a small car-rental MDP."""
    mdp = CarRentalMDP(cap=cap, gamma=gamma, poisson_cap=8)
    V, pi, deltas = mdp.value_iteration(tol=1e-2, max_iter=80)
    return V, pi, len(deltas)
