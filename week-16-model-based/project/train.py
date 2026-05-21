"""Dreamer-lite training loop on CartPole-v1.

Target metric: mean episode return >= 50 within ~30s on a single GPU
(or ~2-3 min on CPU). CartPole's max is 500; we deliberately stop short
so the test stays fast.

The pipeline:
  1. Collect a few episodes with a random/actor policy into a replay buffer.
  2. World-model update: sample sequence batches, run RSSM in observe mode,
     minimise reconstruction + reward + continue + KL.
  3. Behaviour update: from each sampled state, imagine H steps under the
     actor; compute the lambda-return; update actor (REINFORCE with
     baseline) and critic (regress to lambda-return).
  4. Repeat. Acting in the real env uses argmax of the actor head.
"""
from __future__ import annotations
import os
import sys
import time
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.seed import seed_everything

from .world_model import RSSM, RSSMState, kl_gaussian
from .agent import DreamerLite


def _onehot(a: int, nA: int) -> np.ndarray:
    v = np.zeros(nA, dtype=np.float32); v[a] = 1.0; return v


def _collect_episode(env, policy_fn, max_steps: int = 200) -> dict:
    obs, _ = env.reset()
    obss, acts, rews, conts = [obs.astype(np.float32)], [], [], []
    done = False; steps = 0
    while not done and steps < max_steps:
        a = policy_fn(obs)
        obs, r, term, trunc, _ = env.step(int(a))
        obss.append(obs.astype(np.float32))
        acts.append(int(a)); rews.append(float(r))
        conts.append(0.0 if (term or trunc) else 1.0)
        done = term or trunc
        steps += 1
    return {"obs": np.asarray(obss, dtype=np.float32),
            "acts": np.asarray(acts, dtype=np.int64),
            "rews": np.asarray(rews, dtype=np.float32),
            "conts": np.asarray(conts, dtype=np.float32)}


def _sample_chunks(episodes: List[dict], n_chunks: int, chunk_len: int,
                   rng: np.random.Generator):
    """Sample fixed-length sequence chunks. Pads short episodes by repeating last obs."""
    obs_dim = episodes[0]["obs"].shape[-1]
    L = chunk_len
    O = np.zeros((n_chunks, L + 1, obs_dim), dtype=np.float32)
    A = np.zeros((n_chunks, L), dtype=np.int64)
    R = np.zeros((n_chunks, L), dtype=np.float32)
    C = np.zeros((n_chunks, L), dtype=np.float32)
    for i in range(n_chunks):
        ep = episodes[int(rng.integers(0, len(episodes)))]
        T = len(ep["acts"])
        if T >= L:
            j = int(rng.integers(0, T - L + 1))
            O[i] = ep["obs"][j:j + L + 1]
            A[i] = ep["acts"][j:j + L]
            R[i] = ep["rews"][j:j + L]
            C[i] = ep["conts"][j:j + L]
        else:
            # pad: repeat the last observation; rewards and conts = 0.
            O[i, :T + 1] = ep["obs"]
            O[i, T + 1:] = ep["obs"][-1]
            A[i, :T] = ep["acts"]
            R[i, :T] = ep["rews"]
            C[i, :T] = ep["conts"]
    return O, A, R, C


def _world_model_update(rssm: RSSM, opt: torch.optim.Optimizer, O, A, R, C,
                        nA: int, kl_beta: float, free_nats: float):
    """One world-model gradient step. Returns dict of scalar losses."""
    device = next(rssm.parameters()).device
    B, Lp1, obs_dim = O.shape
    L = Lp1 - 1
    O_t = torch.from_numpy(O).to(device)
    A_oh = F.one_hot(torch.from_numpy(A), nA).float().to(device)
    R_t = torch.from_numpy(R).to(device)
    C_t = torch.from_numpy(C).to(device)

    state = rssm.initial(B, device)
    recon_loss = 0.0; rew_loss = 0.0; cont_loss = 0.0; kl_loss = 0.0
    for t in range(L):
        target_obs = O_t[:, t + 1]
        (state, prior_mu, prior_std, post_mu, post_std,
         recon, r_pred, c_pred) = rssm.observe_step(state, A_oh[:, t], target_obs)
        recon_loss = recon_loss + ((recon - target_obs) ** 2).sum(dim=-1).mean()
        rew_loss = rew_loss + ((r_pred - R_t[:, t]) ** 2).mean()
        cont_loss = cont_loss + F.binary_cross_entropy(c_pred, C_t[:, t])
        kl = kl_gaussian(post_mu, post_std, prior_mu, prior_std)
        kl_clipped = torch.clamp(kl, min=free_nats).mean()
        kl_loss = kl_loss + kl_clipped

    recon_loss = recon_loss / L; rew_loss = rew_loss / L
    cont_loss = cont_loss / L; kl_loss = kl_loss / L
    total = recon_loss + rew_loss + cont_loss + kl_beta * kl_loss

    opt.zero_grad(); total.backward()
    torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
    opt.step()
    return {"recon": float(recon_loss), "rew": float(rew_loss),
            "cont": float(cont_loss), "kl": float(kl_loss),
            "total_wm": float(total)}


def _imagine_from_states(rssm: RSSM, agent: DreamerLite, init_state: RSSMState,
                         horizon: int, n_actions: int):
    """Roll the actor for `horizon` steps in the world model. Returns lists of
    features, log_probs, rewards, continues."""
    feats, lps, rs, cs = [], [], [], []
    state = init_state
    for _ in range(horizon):
        feat = torch.cat([state.h, state.s], dim=-1)
        dist = agent.actor(feat)
        a = dist.sample()
        lp = dist.log_prob(a)
        a_oh = F.one_hot(a, n_actions).float()
        state, r, c = rssm.imagine_step(state, a_oh)
        feats.append(feat); lps.append(lp); rs.append(r); cs.append(c)
    # Append final feat for bootstrap value.
    feats.append(torch.cat([state.h, state.s], dim=-1))
    return feats, lps, rs, cs


def _lambda_return(rewards, values, conts, gamma: float, lam: float):
    """Compute lambda-returns G_t^lambda along the imagination horizon.
    rewards: list length H of (B,) tensors; values: length H+1; conts: length H."""
    H = len(rewards)
    G = values[-1]
    returns = [None] * H
    for t in reversed(range(H)):
        td = rewards[t] + gamma * conts[t] * ((1 - lam) * values[t + 1] + lam * G)
        G = td
        returns[t] = G
    return torch.stack(returns, dim=0)  # (H, B)


def evaluate(env, agent: DreamerLite, rssm: RSSM, n_episodes: int = 5,
             max_steps: int = 200) -> float:
    device = next(rssm.parameters()).device
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = rssm.initial(1, device)
        G = 0.0
        for t in range(max_steps):
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
            # Posterior step with zero "previous action" (or sampled). We use the
            # observe path with the last action one-hot zero -- acceptable for eval.
            a_oh = torch.zeros(1, agent.n_actions, device=device)
            with torch.no_grad():
                state, _, _, _, _, _, _, _ = rssm.observe_step(state, a_oh, obs_t)
                feat = torch.cat([state.h, state.s], dim=-1)
                a = int(agent.actor(feat).logits.argmax(dim=-1).item())
            obs, r, term, trunc, _ = env.step(a)
            G += r
            if term or trunc:
                break
        returns.append(G)
    return float(np.mean(returns))


def train(seed: int = 0, n_iters: int = 25, episodes_per_iter: int = 2,
          chunk_len: int = 16, batch_size: int = 16, imagine_horizon: int = 10,
          gamma: float = 0.99, lam: float = 0.95, kl_beta: float = 1.0,
          free_nats: float = 1.0, wm_lr: float = 6e-4, ac_lr: float = 4e-4,
          device: str | None = None, max_steps: int = 200) -> Dict[str, Any]:
    seed_everything(seed)
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    env.reset(seed=seed); eval_env.reset(seed=seed + 100)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    obs_dim = env.observation_space.shape[0]
    nA = env.action_space.n

    rssm = RSSM(obs_dim=obs_dim, act_dim=nA, stoch_dim=16, det_dim=64, hidden=128).to(device_t)
    agent = DreamerLite(rssm, n_actions=nA, hidden=128, device=str(device_t))
    wm_opt = torch.optim.Adam(rssm.parameters(), lr=wm_lr)
    ac_opt = torch.optim.Adam(list(agent.actor.parameters()) + list(agent.critic.parameters()),
                              lr=ac_lr)
    rng = np.random.default_rng(seed)

    # Seed the buffer with random episodes.
    episodes = []
    for _ in range(4):
        ep = _collect_episode(env, lambda o: rng.integers(0, nA), max_steps=max_steps)
        episodes.append(ep)

    eval_curve = []
    t0 = time.time()
    for it in range(n_iters):
        # 1) Collect with current actor (epsilon-greedy mixed with random).
        def actor_policy(obs):
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device_t)
            state = rssm.initial(1, device_t)
            a_oh = torch.zeros(1, nA, device=device_t)
            with torch.no_grad():
                state, *_ = rssm.observe_step(state, a_oh, obs_t)
                feat = torch.cat([state.h, state.s], dim=-1)
                dist = agent.actor(feat)
            if rng.random() < 0.2:
                return int(rng.integers(0, nA))
            return int(dist.sample().item())
        for _ in range(episodes_per_iter):
            episodes.append(_collect_episode(env, actor_policy, max_steps=max_steps))
            if len(episodes) > 50:
                episodes = episodes[-50:]

        # 2) World model update (a few steps per iter).
        wm_logs = []
        for _ in range(8):
            O, A, R, C = _sample_chunks(episodes, batch_size, chunk_len, rng)
            log = _world_model_update(rssm, wm_opt, O, A, R, C, nA, kl_beta, free_nats)
            wm_logs.append(log)

        # 3) Behaviour update: roll out imagination from posterior states.
        for _ in range(4):
            O, A, R, C = _sample_chunks(episodes, batch_size, chunk_len, rng)
            O_t = torch.from_numpy(O).to(device_t)
            A_oh = F.one_hot(torch.from_numpy(A), nA).float().to(device_t)
            with torch.no_grad():
                state = rssm.initial(batch_size, device_t)
                for t in range(chunk_len):
                    state, *_ = rssm.observe_step(state, A_oh[:, t], O_t[:, t + 1])
            feats, lps, rs, cs = _imagine_from_states(rssm, agent, state, imagine_horizon, nA)
            vals = [agent.critic(f) for f in feats]
            returns = _lambda_return(rs, vals, cs, gamma, lam)  # (H, B)
            adv = (returns.detach() - torch.stack(vals[:-1], dim=0))
            actor_loss = -(torch.stack(lps, dim=0) * adv.detach()).mean()
            critic_targets = returns.detach()
            critic_pred = torch.stack(vals[:-1], dim=0)
            critic_loss = F.mse_loss(critic_pred, critic_targets)

            ac_opt.zero_grad()
            (actor_loss + critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.actor.parameters()) + list(agent.critic.parameters()), 100.0)
            ac_opt.step()

        # 4) Evaluate.
        ret = evaluate(eval_env, agent, rssm, n_episodes=3, max_steps=max_steps)
        eval_curve.append((it, ret))

    return {
        "eval_curve": np.asarray(eval_curve, dtype=np.float32),
        "rssm": rssm, "agent": agent,
        "elapsed_sec": time.time() - t0,
    }


if __name__ == "__main__":
    out = train(seed=0)
    print("eval curve:", out["eval_curve"])
    print("elapsed s:", out["elapsed_sec"])
