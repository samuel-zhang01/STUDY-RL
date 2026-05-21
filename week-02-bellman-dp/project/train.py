"""End-to-end training driver for week 2.

Target: value iteration on a 5x5 deterministic GridWorld converges in
fewer than 200 sweeps to a sup-norm tolerance of 1e-8. Total runtime
< 60 s on CPU.
"""
from __future__ import annotations
from typing import Dict
import numpy as np

from common.gridworld import GridWorld
from common.tabular import value_iteration, policy_iteration

from .carrental import CarRentalMDP


def train(seed: int = 0) -> Dict[str, float]:
    out: Dict[str, float] = {}
    # Deterministic GridWorld
    env_det = GridWorld(rows=5, cols=5, start=(0, 0), goals=((4, 4),))
    V_vi, pi_vi, deltas = value_iteration(env_det, gamma=0.95)
    out["det_vi_sweeps"] = float(len(deltas))
    out["det_vi_V_start"] = float(V_vi[env_det.s2idx((0, 0))])
    # Policy iteration
    V_pi, pi_pi, n_sweeps = policy_iteration(env_det, gamma=0.95)
    out["det_pi_sweeps"] = float(n_sweeps)
    # Stochastic GridWorld
    env_stoch = GridWorld(rows=5, cols=5, start=(0, 0), goals=((4, 4),),
                          slip_prob=0.2)
    V_s, _, _ = value_iteration(env_stoch, gamma=0.95)
    out["stoch_vi_V_start"] = float(V_s[env_stoch.s2idx((0, 0))])
    # Car rental, small instance
    mdp = CarRentalMDP(cap=5, gamma=0.9, poisson_cap=8)
    V_c, pi_c, deltas_c = mdp.value_iteration(tol=1e-2, max_iter=80)
    out["car_rental_V_at_5_5"] = float(V_c[mdp.s2idx((5, 5))])
    out["car_rental_iters"] = float(len(deltas_c))
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    o = train()
    dt = time.time() - t0
    print(f"finished in {dt:.1f}s")
    for k, v in o.items():
        print(f"  {k:>22s}: {v:.4f}")
