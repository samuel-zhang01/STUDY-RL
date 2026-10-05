"""Jack's car-rental MDP (Sutton & Barto, Example 4.2), trimmed.

State: (cars at location 1, cars at location 2), each capped at ``cap``.
Action: number of cars moved overnight from location 1 to location 2, in
[-cap, +cap]; positive = 1 -> 2; negative = 2 -> 1. Each move costs $2.

Each day, requests at location i are Poisson(lambda_req_i) and returns are
Poisson(lambda_ret_i). A successful rental earns $10. We model the dynamics
exactly for the capped inventory via Poisson PMFs whose last bin absorbs
the tail. The bin horizon is at least the inventory capacity, so larger
request/return counts cannot change any capped outcome.

For tractability the default is ``cap=10`` (Sutton uses 20), which keeps the
state space to 121 states and a transition tensor under ~50 MB.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np


def _poisson_pmf(lam: float, cap: int) -> np.ndarray:
    """Truncated Poisson PMF on {0,...,cap}, last bin absorbing the tail."""
    pmf = np.zeros(cap + 1)
    for k in range(cap):
        pmf[k] = math.exp(-lam) * lam ** k / math.factorial(k)
    pmf[cap] = 1.0 - pmf[:cap].sum()
    return pmf


@dataclass
class CarRentalMDP:
    cap: int = 10
    move_cost: float = 2.0
    rental_reward: float = 10.0
    lambda_req: Tuple[float, float] = (3.0, 4.0)
    lambda_ret: Tuple[float, float] = (3.0, 2.0)
    poisson_cap: int = 11  # truncation horizon for Poisson tail
    gamma: float = 0.9

    def __post_init__(self) -> None:
        self.poisson_cap = max(self.poisson_cap, self.cap)
        self.req_pmf = [_poisson_pmf(self.lambda_req[i], self.poisson_cap) for i in (0, 1)]
        self.ret_pmf = [_poisson_pmf(self.lambda_ret[i], self.poisson_cap) for i in (0, 1)]
        self._build_transition()

    # ------------------------------------------------------------------
    @property
    def nS(self) -> int:
        return (self.cap + 1) ** 2

    @property
    def actions(self) -> List[int]:
        return list(range(-self.cap, self.cap + 1))

    def s2idx(self, s: Tuple[int, int]) -> int:
        return s[0] * (self.cap + 1) + s[1]

    def idx2s(self, idx: int) -> Tuple[int, int]:
        return (idx // (self.cap + 1), idx % (self.cap + 1))

    def _build_transition(self) -> None:
        """Compute the expected reward R(s,a) and transition matrix P(s'|s,a)
        for every state-action pair. Marginalises out the Poisson requests and
        returns analytically (no Monte Carlo).
        """
        cap = self.cap
        nS = self.nS
        # We marginalise demand and returns per location independently.
        # next_dist1[n][n']: P(n' cars at loc1 tomorrow | n today)
        next_dist = [np.zeros((cap + 1, cap + 1)) for _ in (0, 1)]
        exp_req = [np.zeros(cap + 1) for _ in (0, 1)]
        for loc in (0, 1):
            req = self.req_pmf[loc]; ret = self.ret_pmf[loc]
            for n in range(cap + 1):
                # actual rentals = min(req, n_after_move)
                for r in range(self.poisson_cap + 1):
                    actual = min(r, n)
                    exp_req[loc][n] += req[r] * actual
                    remaining = n - actual
                    for ret_n in range(self.poisson_cap + 1):
                        npr = min(cap, remaining + ret_n)
                        next_dist[loc][n, npr] += req[r] * ret[ret_n]
        self.next_dist = next_dist
        self.exp_req = exp_req

        self.actions_list = self.actions
        self.nA = len(self.actions_list)

    def expected_reward_and_next(self, s: Tuple[int, int], a: int
                                 ) -> Tuple[float, np.ndarray]:
        """Return (expected immediate reward, transition pmf over next states)."""
        n1, n2 = s
        # Feasibility requires both available cars and destination space.
        # A transfer must not silently discard cars at a full destination.
        lower = -min(n2, self.cap - n1)
        upper = min(n1, self.cap - n2)
        a_eff = max(lower, min(upper, a))
        n1m, n2m = n1 - a_eff, n2 + a_eff
        # Clip at capacity
        n1m = min(self.cap, max(0, n1m))
        n2m = min(self.cap, max(0, n2m))
        move_r = -self.move_cost * abs(a_eff)
        rent_r = self.rental_reward * (self.exp_req[0][n1m] + self.exp_req[1][n2m])
        nd1 = self.next_dist[0][n1m]
        nd2 = self.next_dist[1][n2m]
        # outer product over (n1', n2'), flattened
        joint = np.outer(nd1, nd2).ravel()
        return float(move_r + rent_r), joint

    def value_iteration(self, tol: float = 1e-3, max_iter: int = 200
                        ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        V = np.zeros(self.nS)
        deltas: List[float] = []
        for _ in range(max_iter):
            V_new = np.zeros_like(V)
            for s_idx in range(self.nS):
                s = self.idx2s(s_idx)
                best = -np.inf
                for a in self.actions_list:
                    r, p_next = self.expected_reward_and_next(s, a)
                    q = r + self.gamma * float(p_next @ V)
                    if q > best:
                        best = q
                V_new[s_idx] = best
            d = float(np.max(np.abs(V_new - V)))
            deltas.append(d)
            V = V_new
            if d < tol:
                break
        pi = np.zeros(self.nS, dtype=int)
        for s_idx in range(self.nS):
            s = self.idx2s(s_idx)
            best, best_a = -np.inf, 0
            for ai, a in enumerate(self.actions_list):
                r, p_next = self.expected_reward_and_next(s, a)
                q = r + self.gamma * float(p_next @ V)
                if q > best:
                    best, best_a = q, ai
            pi[s_idx] = best_a
        return V, pi, deltas
