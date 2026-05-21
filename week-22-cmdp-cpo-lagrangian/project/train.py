"""Training entry point.

`train(seed=0)` runs PPO-Lagrangian on `LavaGridWorld` for ~80 dual updates and
returns the history dict. Target: median return >= 0.5 (i.e. the policy reaches
the goal at least half the time, more or less) while keeping discounted cost
<= cost_limit + 1 (mild overshoot tolerated).

Run as
    python -m week_22_cmdp_cpo_lagrangian.project.train
or import:
    from week-22-cmdp-cpo-lagrangian.project.train import train
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from .env import LavaGridWorld
from .agent import PPOLagrangian, PPOLagrangianConfig, RolloutBuffer


def _collect(env: LavaGridWorld, agent: PPOLagrangian, steps: int, gamma: float):
    buf = RolloutBuffer()
    ep_returns: List[float] = []
    ep_costs: List[float] = []
    ep_R = 0.0
    ep_C = 0.0
    disc = 1.0
    obs_idx = env.reset()
    obs = env.one_hot_obs(obs_idx)
    for _ in range(steps):
        with torch.no_grad():
            a, lp, v, vc = agent.net.act(torch.as_tensor(obs).unsqueeze(0))
        nobs_idx, r, c, done, _ = env.step(a)
        buf.push(obs.copy(), a, lp, r, c, v, vc, done)
        ep_R += r
        ep_C += disc * c
        disc *= gamma
        obs = env.one_hot_obs(nobs_idx)
        if done:
            ep_returns.append(ep_R)
            ep_costs.append(ep_C)
            ep_R = 0.0
            ep_C = 0.0
            disc = 1.0
            obs_idx = env.reset()
            obs = env.one_hot_obs(obs_idx)
    # bootstrap value at the tail
    with torch.no_grad():
        _, v_last, vc_last = agent.net(torch.as_tensor(obs).unsqueeze(0))
        last_v = float(v_last.item())
        last_vc = float(vc_last.item())
    return buf, last_v, last_vc, ep_returns, ep_costs


def train(seed: int = 0, n_updates: int = 80, steps_per_update: int = 512,
          cost_limit: float = 1.5, verbose: bool = False) -> Dict[str, list]:
    """PPO-Lagrangian on the lava gridworld. Returns the agent's history dict."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = LavaGridWorld()
    env.reset(seed=seed)

    cfg = PPOLagrangianConfig(cost_limit=cost_limit)
    agent = PPOLagrangian(obs_dim=env.n_states, n_actions=env.n_actions, cfg=cfg)

    for it in range(n_updates):
        buf, last_v, last_vc, ep_rets, ep_costs = _collect(
            env, agent, steps_per_update, cfg.gamma)
        agent.update(buf, last_v, last_vc, ep_rets, ep_costs)
        if verbose and (it % 10 == 0 or it == n_updates - 1):
            print(f"it={it:3d}  R={agent.history['return'][-1]:+.3f}  "
                  f"C={agent.history['cost'][-1]:.3f}  lam={agent.lam:.3f}")
    return agent.history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=80)
    ap.add_argument("--cost-limit", type=float, default=1.5)
    args = ap.parse_args()
    h = train(seed=args.seed, n_updates=args.updates, cost_limit=args.cost_limit, verbose=True)
    print("final R:", h["return"][-1], "final C:", h["cost"][-1], "final lam:", h["lambda"][-1])


if __name__ == "__main__":  # pragma: no cover
    main()
