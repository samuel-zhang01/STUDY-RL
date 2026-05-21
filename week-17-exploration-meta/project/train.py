"""Driver: train RND and MAML, return summary.

Target metric: with RND enabled, the agent finds the goal at least 2x more often
than the no-RND baseline within 200 episodes; MAML post-adapt reward exceeds
0.7 within 200 outer steps.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from .rnd import train_rnd_on_gridworld
from .maml import train_maml


def train(seed: int = 0, n_episodes_rnd: int = 200, n_outer_maml: int = 200) -> dict:
    rnd_off = train_rnd_on_gridworld(seed=seed, n_episodes=n_episodes_rnd, use_rnd=False)
    rnd_on = train_rnd_on_gridworld(seed=seed, n_episodes=n_episodes_rnd, use_rnd=True)
    maml_out = train_maml(seed=seed, n_outer=n_outer_maml)
    return {"rnd_off": rnd_off, "rnd_on": rnd_on, "maml": maml_out}


if __name__ == "__main__":
    out = train()
    print("RND off, goal hits:", out["rnd_off"]["goal_hits"])
    print("RND on,  goal hits:", out["rnd_on"]["goal_hits"])
    print("MAML last reward:", float(out["maml"]["rewards_curve"][-1]))
