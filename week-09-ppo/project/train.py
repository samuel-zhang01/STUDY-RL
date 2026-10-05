"""Train PPO on BipedalWalker-v3, with most of the 37 implementation details.

Target metric: mean episode return trends upward (final - initial >= +30) within
60s on CPU, and to mean return >= 100 within 5min on a single consumer GPU.

The training loop tracks every detail with a comment of the form '# detail #NN'
so the workbook can audit which details are present.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from common.seed import seed_everything
from common.rollouts import (
    sync_vector_env, transition_next_observations, generalized_advantage_estimate,
)

from .agent import PPOAgent


@dataclass
class TrainOutput:
    episode_returns: list[float]
    update_metrics: list[dict] = field(default_factory=list)
    details_used: dict[int, bool] = field(default_factory=dict)


def _make_vec_env(env_id: str, num_envs: int, seed: int):
    import gymnasium as gym

    def thunk(i: int):
        def _f():
            e = gym.make(env_id)
            e.reset(seed=seed + i)
            return e
        return _f

    # detail #1: vectorised env
    return sync_vector_env([thunk(i) for i in range(num_envs)])


def _compute_gae(rewards, values, dones, last_value, gamma, lam, *,
                 next_values=None, terminals=None):
    return generalized_advantage_estimate(
        rewards, values, dones, last_value, gamma, lam,
        next_values=next_values, terminals=terminals,
    )


def train(
    *,
    seed: int = 0,
    env_id: str = "BipedalWalker-v3",
    num_envs: int = 4,
    rollout_steps: int = 256,
    n_updates: int = 100,
    n_epochs: int = 4,                 # detail #31
    minibatch_size: int = 64,
    gamma: float = 0.99,
    lam: float = 0.95,                 # detail #5
    lr: float = 3e-4,
    clip_eps: float = 0.2,             # detail #7
    target_kl: float = 0.02,           # detail #11
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    anneal_lr: bool = True,            # detail #4
    use_value_clip: bool = True,       # detail #8
    use_advantage_norm: bool = True,   # detail #6
    early_stop_kl: bool = True,        # detail #11
    verbose: bool = False,
) -> TrainOutput:
    seed_everything(seed)
    envs = _make_vec_env(env_id, num_envs, seed)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = PPOAgent.build(
        obs_dim, act_dim, lr=lr,
        clip_eps=clip_eps, value_coef=value_coef,
        entropy_coef=entropy_coef, target_kl=target_kl,
    )

    obs, _ = envs.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)

    episode_returns: list[float] = []
    update_metrics: list[dict] = []
    current_returns = np.zeros(num_envs, dtype=np.float64)

    base_lr = lr
    initial_window_mean: float | None = None
    win = deque(maxlen=20)

    for it in range(n_updates):
        # detail #4: linear LR anneal.
        if anneal_lr:
            frac = max(1e-3, 1.0 - it / max(1, n_updates))
            for g in agent.opt.param_groups:
                g["lr"] = base_lr * frac

        # ---- rollout ------------------------------------------------------
        all_obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        all_acts = np.zeros((rollout_steps, num_envs, act_dim), dtype=np.float32)
        all_logp = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        all_vals = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        all_rews = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        all_dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        all_terminals = np.zeros_like(all_dones)
        all_next_vals = np.zeros_like(all_vals)

        for t in range(rollout_steps):
            a, lp, v = agent.act(obs)
            # Preserve Gaussian samples for PPO ratios; clip only at the
            # environment boundary, using this environment's actual bounds.
            env_actions = np.clip(a, envs.single_action_space.low, envs.single_action_space.high)
            nxt, r, term, trunc, infos = envs.step(env_actions)
            done = np.logical_or(term, trunc)
            bootstrap_obs = transition_next_observations(nxt, infos, done)
            with torch.no_grad():
                next_v = agent.critic(torch.as_tensor(bootstrap_obs, dtype=torch.float32)).cpu().numpy()
            all_obs[t] = obs
            all_acts[t] = a
            all_logp[t] = lp
            all_vals[t] = v
            all_rews[t] = r
            all_dones[t] = done.astype(np.float32)
            all_terminals[t] = term.astype(np.float32)
            all_next_vals[t] = next_v

            current_returns += r
            for i in range(num_envs):
                if done[i]:
                    episode_returns.append(float(current_returns[i]))
                    win.append(float(current_returns[i]))
                    current_returns[i] = 0.0
            obs = np.asarray(nxt, dtype=np.float32)

        with torch.no_grad():
            last_vals = agent.critic(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()

        # ---- GAE per env --------------------------------------------------
        advs = np.zeros_like(all_rews)
        rets = np.zeros_like(all_rews)
        for i in range(num_envs):
            a, r = _compute_gae(all_rews[:, i], all_vals[:, i], all_dones[:, i],
                                float(last_vals[i]), gamma, lam,
                                next_values=all_next_vals[:, i], terminals=all_terminals[:, i])
            advs[:, i] = a; rets[:, i] = r

        if initial_window_mean is None and len(win) > 0:
            initial_window_mean = float(np.mean(win))

        # ---- flatten + epochs of mini-batch SGD ---------------------------
        N = rollout_steps * num_envs
        idx_all = np.arange(N)

        obs_flat = torch.as_tensor(all_obs.reshape(N, obs_dim), dtype=torch.float32)
        act_flat = torch.as_tensor(all_acts.reshape(N, act_dim), dtype=torch.float32)
        logp_old = torch.as_tensor(all_logp.reshape(N), dtype=torch.float32)
        v_old = torch.as_tensor(all_vals.reshape(N), dtype=torch.float32)
        adv_flat = torch.as_tensor(advs.reshape(N), dtype=torch.float32)
        ret_flat = torch.as_tensor(rets.reshape(N), dtype=torch.float32)

        ep_metrics = {"pi_loss": [], "v_loss": [], "entropy": [], "approx_kl": []}

        for _ in range(n_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, N, minibatch_size):
                idx = idx_all[start:start + minibatch_size]
                m = agent.update_minibatch(
                    obs_flat[idx], act_flat[idx], logp_old[idx],
                    adv_flat[idx], ret_flat[idx], v_old[idx],
                    use_value_clip=use_value_clip, use_advantage_norm=use_advantage_norm,
                )
                for k, v in m.items():
                    ep_metrics[k].append(v)
            # detail #11: KL early-stop
            if early_stop_kl and ep_metrics["approx_kl"]:
                if np.mean(ep_metrics["approx_kl"][-N // minibatch_size:]) > 1.5 * target_kl:
                    break

        update_metrics.append({k: float(np.mean(vs)) if vs else 0.0
                                for k, vs in ep_metrics.items()})

        if verbose and it % 10 == 0:
            print(f"upd {it:4d}  recent {np.mean(win) if win else float('nan'):7.1f}  "
                  f"pi {update_metrics[-1]['pi_loss']:+.3f}  "
                  f"kl {update_metrics[-1]['approx_kl']:+.4f}")

    envs.close()

    details_used = {
        1: True, 2: True, 3: True, 4: anneal_lr, 5: True, 6: use_advantage_norm,
        7: True, 8: use_value_clip, 9: True, 10: True, 11: early_stop_kl,
        15: True, 17: True, 18: True, 28: True, 29: True, 31: True, 32: True,
    }

    return TrainOutput(
        episode_returns=episode_returns,
        update_metrics=update_metrics,
        details_used=details_used,
    )


if __name__ == "__main__":  # pragma: no cover
    out = train(verbose=True)
    if out.episode_returns:
        print("final mean return:", np.mean(out.episode_returns[-30:]))
