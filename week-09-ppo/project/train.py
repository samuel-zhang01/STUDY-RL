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
    return gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])


def _compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in range(T - 1, -1, -1):
        v_next = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * v_next * nonterm - values[t]
        last_adv = delta + gamma * lam * nonterm * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


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

        for t in range(rollout_steps):
            a, lp, v = agent.act(obs)
            nxt, r, term, trunc, _ = envs.step(a)
            done = np.logical_or(term, trunc)
            all_obs[t] = obs
            all_acts[t] = a
            all_logp[t] = lp
            all_vals[t] = v
            all_rews[t] = r
            all_dones[t] = done.astype(np.float32)

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
                                float(last_vals[i]), gamma, lam)
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
