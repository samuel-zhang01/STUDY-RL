"""End-to-end training for Week 3.

Target: on-policy MC control on Blackjack reaches average return >= -0.06
after 30k episodes (Sutton & Barto's figure 5.2 reports ~-0.05 at infinity;
30k is enough to clearly beat the all-stick / all-hit baselines which both
sit below -0.10). Runtime < 60s on CPU.
"""
from __future__ import annotations
from typing import Dict
import numpy as np

from .solutions import (
    exercise_3_blackjack_mc_control,
    exercise_7_offpolicy_recruitment,
)


def train(seed: int = 0, n_episodes: int = 30_000) -> Dict[str, float]:
    Q, pi, returns = exercise_3_blackjack_mc_control(
        n_episodes=n_episodes, eps=0.1, seed=seed
    )
    # Average return on the final 5k episodes.
    last_avg = float(np.mean(returns[-5_000:]))
    # Off-policy estimator on recruitment funnel.
    v_ois, v_wis, v_oracle = exercise_7_offpolicy_recruitment(n_episodes=3000, seed=seed)
    return {
        "blackjack_avg_return_last_5k": last_avg,
        "n_state_action_visited": float(len({k for k in Q.keys()})),
        "recruitment_V_ois": v_ois,
        "recruitment_V_wis": v_wis,
        "recruitment_V_oracle": v_oracle,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = train()
    dt = time.time() - t0
    print(f"finished in {dt:.1f}s")
    for k, v in out.items():
        print(f"  {k:>32s}: {v:.4f}")
