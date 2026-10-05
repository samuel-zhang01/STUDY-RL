"""Reference solutions for week-5 workbook exercises."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Exercise 1: gather Q(s,a) from a Q-network ------------------------

def exercise_1_gather_q(q_all: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Given Q(s,*) of shape (B, A) and integer actions of shape (B,),
    return Q(s,a) of shape (B,)."""
    return q_all.gather(1, actions.long().unsqueeze(1)).squeeze(1)


# ----- Exercise 2: vanilla DQN target ------------------------------------

def exercise_2_dqn_target(reward: torch.Tensor, done: torch.Tensor,
                          q_target_next: torch.Tensor, gamma: float) -> torch.Tensor:
    """target = r + (1-done) * gamma * max_a' Q_target(s', a').

    reward, done: shape (B,)  ; q_target_next: shape (B, A).
    """
    q_max = q_target_next.max(dim=1).values
    return reward + (1.0 - done.float()) * gamma * q_max


# ----- Exercise 3: Polyak soft update ------------------------------------

def exercise_3_polyak(target: nn.Module, online: nn.Module, tau: float) -> None:
    """In-place: theta_target <- (1-tau) theta_target + tau theta_online."""
    with torch.no_grad():
        for p, pt in zip(online.parameters(), target.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)


# ----- Exercise 4: epsilon-greedy from a Q-network -----------------------

def exercise_4_eps_greedy(q_values: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    """q_values: shape (A,). Return chosen action."""
    if rng.random() < eps:
        return int(rng.integers(0, q_values.shape[0]))
    return int(np.argmax(q_values))


# ----- Exercise 5: TD error for prioritised replay ----------------------

def exercise_5_td_error(q_sa: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element TD error |delta_i| used as PER priority. Shape (B,)."""
    return (target - q_sa).detach().abs()


# ----- Exercise 6: Huber loss --------------------------------------------

def exercise_6_huber(pred: torch.Tensor, target: torch.Tensor,
                     delta: float = 1.0) -> torch.Tensor:
    """Mean smooth-L1 loss with transition threshold delta.

    Smooth-L1 equals Huber/delta; the two coincide when delta is 1.
    """
    return F.smooth_l1_loss(pred, target, reduction="mean", beta=delta)


# ----- Exercise 7: deadly-triad divergence demo --------------------------

def exercise_7_baird_divergence(n_iters: int = 200, alpha: float = 0.1,
                                gamma: float = 0.99) -> np.ndarray:
    """Seven-state Baird counterexample with importance-weighted TD(0).

    Target chooses solid (next state 6). Behaviour chooses solid with
    probability 1/7 and dashed (uniform upper state) with probability 6/7.
    The importance ratio is 7 on solid actions and 0 on dashed actions.
    Features have unit norm; all rewards are zero. Returns parameter norms.
    """
    # 7-state Baird's counterexample (small version)
    n_states = 7
    n_features = 8
    # feature matrix Phi (state x feature)
    Phi = np.zeros((n_states, n_features))
    for i in range(n_states - 1):
        Phi[i, i] = 2.0
        Phi[i, 7] = 1.0
    Phi[n_states - 1, 6] = 1.0
    Phi[n_states - 1, 7] = 2.0
    Phi /= np.sqrt(5.0)
    w = np.ones(n_features)
    w[6] = 10.0
    norms = []
    rng = np.random.default_rng(0)
    s = 0
    for _ in range(n_iters):
        solid = rng.random() < 1.0 / n_states
        sp = n_states - 1 if solid else int(rng.integers(0, n_states - 1))
        v_s = Phi[s] @ w
        v_sp = Phi[sp] @ w
        delta = 0.0 + gamma * v_sp - v_s
        # importance-weighted semi-gradient (off-policy)
        rho = float(n_states) if solid else 0.0
        w = w + alpha * rho * delta * Phi[s]
        s = sp
        norms.append(float(np.linalg.norm(w)))
    return np.asarray(norms)


# ----- Exercise 8: smoke training pass -----------------------------------

def exercise_8_smoke_dqn(n_steps: int = 2000, seed: int = 0):
    """Train DQN for a handful of steps and return the train log."""
    from .train import train
    return train(seed=seed, n_steps=n_steps, eps_decay_steps=1500,
                 warmup_steps=200, buffer_size=5000, batch_size=64,
                 log_every=100)


# ----- Exercise 9 (★★★): visualise action-values across an episode ------

def exercise_9_action_value_trace(agent, env, max_steps: int = 200,
                                  seed: int = 0):
    """Roll out a deterministic policy and record Q(s, a) at every step.

    Returns (states, q_values_per_step, actions, rewards).
    """
    s, _ = env.reset(seed=seed)
    qs, acts, rews, states = [], [], [], []
    for _ in range(max_steps):
        states.append(np.asarray(s, dtype=np.float32))
        x = torch.as_tensor(s, dtype=torch.float32,
                            device=agent.cfg.device).unsqueeze(0)
        with torch.no_grad():
            q = agent.q(x).cpu().numpy().squeeze(0)
        qs.append(q)
        a = int(np.argmax(q))
        acts.append(a)
        s, r, term, trunc, _ = env.step(a)
        rews.append(float(r))
        if term or trunc:
            break
    return (np.asarray(states), np.asarray(qs), np.asarray(acts),
            np.asarray(rews))
