"""Train the toy Decision Transformer on the synthetic GridWorld dataset.

Target metric: training cross-entropy loss drops by at least 40% from epoch 0
and conditional rollouts with high target return achieve mean return > 0
within 60 seconds on CPU.
"""
from __future__ import annotations
import os
import sys
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.seed import seed_everything

from .data import build_dt_dataset, sample_minibatch, make_env
from .model import DecisionTransformer


def train(seed: int = 0, n_steps: int = 1500, batch_size: int = 32,
          context_len: int = 8, lr: float = 3e-4, hidden: int = 64,
          n_episodes_dataset: int = 200, device: str | None = None) -> Dict[str, Any]:
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    episodes, K, nS, nA = build_dt_dataset(n_episodes=n_episodes_dataset,
                                           context_len=context_len, seed=seed)
    model = DecisionTransformer(state_dim=nS, n_actions=nA, context_len=K,
                                hidden=hidden, max_timesteps=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    losses = []
    for step in range(n_steps):
        S, A, R, T, M = sample_minibatch(episodes, K, batch_size, rng)
        S_t = torch.from_numpy(S).to(device)
        A_t = torch.from_numpy(A).to(device)
        R_t = torch.from_numpy(R[..., None]).to(device)
        T_clipped = np.clip(T, 0, 255)
        T_t = torch.from_numpy(T_clipped).to(device)
        M_t = torch.from_numpy(M).to(device)

        logits = model(R_t, S_t, A_t, T_t)  # (B, K, nA)
        loss_unmasked = F.cross_entropy(logits.reshape(-1, nA), A_t.reshape(-1),
                                        reduction="none").reshape_as(M_t)
        loss = (loss_unmasked * M_t).sum() / M_t.sum().clamp_min(1.0)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

    # Conditional generation: roll out the policy targeting a *high* return.
    env = make_env(seed=seed)
    eval_returns_high = _rollout(model, env, target_return=3.0, n_eps=50,
                                 context_len=K, device=device, seed=seed + 99)
    eval_returns_low = _rollout(model, env, target_return=-5.0, n_eps=50,
                                context_len=K, device=device, seed=seed + 100)
    return {
        "losses": np.asarray(losses, dtype=np.float32),
        "mean_return_high_target": float(np.mean(eval_returns_high)),
        "mean_return_low_target": float(np.mean(eval_returns_low)),
        "model": model,
    }


def _rollout(model: DecisionTransformer, env, target_return: float,
             n_eps: int, context_len: int, device: str, seed: int) -> list[float]:
    model.eval()
    nS, nA, K = model.state_dim, model.n_actions, context_len
    rng = np.random.default_rng(seed)
    returns = []
    with torch.no_grad():
        for ep in range(n_eps):
            env.reset(seed=int(rng.integers(0, 2**31)))
            s = env.state
            S_hist = np.zeros((1, K, nS), dtype=np.float32)
            A_hist = np.zeros((1, K), dtype=np.int64)
            R_hist = np.zeros((1, K), dtype=np.float32)
            T_hist = np.zeros((1, K), dtype=np.int64)
            R_to_go = float(target_return)
            ret = 0.0
            for t in range(30):
                # Roll the window.
                S_hist = np.roll(S_hist, -1, axis=1)
                A_hist = np.roll(A_hist, -1, axis=1)
                R_hist = np.roll(R_hist, -1, axis=1)
                T_hist = np.roll(T_hist, -1, axis=1)
                S_hist[0, -1] = 0.0; S_hist[0, -1, env.s2idx(s)] = 1.0
                R_hist[0, -1] = R_to_go
                T_hist[0, -1] = min(t, 63)

                S_t = torch.from_numpy(S_hist).to(device)
                A_t = torch.from_numpy(A_hist).to(device)
                R_t = torch.from_numpy(R_hist[..., None]).to(device)
                T_t = torch.from_numpy(T_hist).to(device)
                logits = model(R_t, S_t, A_t, T_t)
                a = int(torch.argmax(logits[0, -1]).item())
                A_hist[0, -1] = a

                s, r, done, _ = env.step(a)
                ret += r
                R_to_go -= r
                if done:
                    break
            returns.append(ret)
    model.train()
    return returns


if __name__ == "__main__":
    out = train(seed=0, n_steps=800)
    print(f"loss[0]={out['losses'][0]:.3f}  loss[-1]={out['losses'][-1]:.3f}")
    print(f"high-target eval return = {out['mean_return_high_target']:.2f}")
    print(f"low-target  eval return = {out['mean_return_low_target']:.2f}")
