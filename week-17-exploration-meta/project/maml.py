"""Tiny MAML on a meta-bandit with two task distributions.

Task distribution:
- Task type A: arm 0 is best with reward ~ N(1, 0.1), others N(0, 0.1).
- Task type B: arm K-1 is best with reward ~ N(1, 0.1), others N(0, 0.1).

The MAML policy is a softmax over arms parametrised by theta in R^K. Inner
adaptation: K steps of REINFORCE-like update on a freshly drawn task. Outer
update: gradient of post-adaptation expected reward w.r.t. theta.

This is the canonical "K-arm bandit" meta-RL toy from Finn et al.\\ 2017.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


class MAMLBandit:
    def __init__(self, n_arms: int = 5):
        self.n_arms = n_arms
        self.theta = torch.zeros(n_arms, requires_grad=True)

    def sample_task(self, rng: np.random.Generator) -> torch.Tensor:
        """Returns a reward-mean vector mu of shape (n_arms,) for a random task."""
        mu = torch.zeros(self.n_arms)
        if rng.random() < 0.5:
            mu[0] = 1.0
        else:
            mu[-1] = 1.0
        return mu

    def policy_probs(self, theta: torch.Tensor) -> torch.Tensor:
        return F.softmax(theta, dim=-1)


def _adapt(theta0: torch.Tensor, mu: torch.Tensor, k_steps: int,
           inner_lr: float, rng: np.random.Generator) -> torch.Tensor:
    """Run k inner gradient steps on a single task.

    Loss: -E[reward] = -(probs * mu).sum().
    We compute *exact* expected reward to make the gradient deterministic at this scale.
    """
    theta = theta0
    for _ in range(k_steps):
        probs = F.softmax(theta, dim=-1)
        loss = -(probs * mu).sum()
        grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
        theta = theta - inner_lr * grad
    return theta


def train_maml(seed: int = 0, n_outer: int = 200, n_tasks_per_outer: int = 8,
               k_inner: int = 3, inner_lr: float = 1.0, outer_lr: float = 0.1,
               n_arms: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    agent = MAMLBandit(n_arms=n_arms)
    opt = torch.optim.SGD([agent.theta], lr=outer_lr)

    rewards_curve = []
    pre_post_gap = []
    for outer in range(n_outer):
        opt.zero_grad()
        total_loss = 0.0
        pre_rs = []
        post_rs = []
        for _ in range(n_tasks_per_outer):
            mu = agent.sample_task(rng)
            # Pre-adaptation expected reward.
            probs0 = F.softmax(agent.theta, dim=-1)
            pre_r = float((probs0.detach() * mu).sum())
            # Inner adapt.
            theta_t = _adapt(agent.theta, mu, k_inner, inner_lr, rng)
            probs_t = F.softmax(theta_t, dim=-1)
            post_loss = -(probs_t * mu).sum()
            total_loss = total_loss + post_loss
            post_rs.append(float(-(post_loss.detach())))
            pre_rs.append(pre_r)
        total_loss = total_loss / n_tasks_per_outer
        total_loss.backward()
        opt.step()
        rewards_curve.append(np.mean(post_rs))
        pre_post_gap.append((np.mean(pre_rs), np.mean(post_rs)))
    return {
        "rewards_curve": np.asarray(rewards_curve, dtype=np.float32),
        "pre_post_gap": np.asarray(pre_post_gap, dtype=np.float32),
        "theta": agent.theta.detach().numpy(),
    }
