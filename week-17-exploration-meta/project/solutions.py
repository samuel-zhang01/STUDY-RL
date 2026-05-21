"""Reference solutions for Week 17 workbook."""
from __future__ import annotations
import numpy as np
import torch

from .rnd import RND, train_rnd_on_gridworld, make_hard_grid
from .maml import MAMLBandit, train_maml, _adapt


def exercise_1_random_target_outputs(seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    rnd = RND(in_dim=10, hidden=16, out_dim=4)
    x = torch.zeros(3, 10)
    with torch.no_grad():
        return rnd.target(x).numpy()


def exercise_2_intrinsic_decreases(seed: int = 0, n_steps: int = 200) -> tuple[float, float]:
    """Train the predictor on a fixed input; intrinsic reward should drop."""
    torch.manual_seed(seed)
    rnd = RND(in_dim=10, hidden=16, out_dim=4)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)
    x = torch.randn(1, 10)
    initial = float(rnd.intrinsic(x).item())
    for _ in range(n_steps):
        loss = rnd.predictor_loss(x)
        opt.zero_grad(); loss.backward(); opt.step()
    final = float(rnd.intrinsic(x).item())
    return initial, final


def exercise_3_train_rnd(seed: int = 0, n_episodes: int = 200, use_rnd: bool = True):
    return train_rnd_on_gridworld(seed=seed, n_episodes=n_episodes, use_rnd=use_rnd)


def exercise_4_rnd_helps(seed: int = 0, n_episodes: int = 200) -> tuple[int, int]:
    off = train_rnd_on_gridworld(seed=seed, n_episodes=n_episodes, use_rnd=False)
    on = train_rnd_on_gridworld(seed=seed, n_episodes=n_episodes, use_rnd=True)
    return off["goal_hits"], on["goal_hits"]


def exercise_5_meta_bandit_sampling(n_samples: int = 100, seed: int = 0) -> np.ndarray:
    """Sample n_samples tasks, return the fraction of type-A tasks."""
    rng = np.random.default_rng(seed)
    agent = MAMLBandit(n_arms=5)
    counts = 0
    for _ in range(n_samples):
        mu = agent.sample_task(rng)
        counts += int(mu[0].item() == 1.0)
    return counts / n_samples


def exercise_6_inner_adaptation_improves(seed: int = 0) -> tuple[float, float]:
    """Single random task: show one inner gradient step increases expected reward."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = MAMLBandit(n_arms=5)
    mu = agent.sample_task(rng)
    probs0 = torch.softmax(agent.theta, dim=-1)
    pre = float((probs0 * mu).sum())
    theta_t = _adapt(agent.theta, mu, k_steps=2, inner_lr=1.0, rng=rng)
    probs_t = torch.softmax(theta_t, dim=-1)
    post = float((probs_t * mu).sum())
    return pre, post


def exercise_7_train_maml(seed: int = 0, n_outer: int = 200):
    return train_maml(seed=seed, n_outer=n_outer)


def exercise_8_in_context_rl_essay() -> str:
    """Meaningful objective: short essay on in-context RL (Algorithm Distillation, DPT)."""
    return (
        "In-context RL turns the meta-learning problem into an autoregressive "
        "sequence modelling task: a transformer is trained on entire learning "
        "histories of different tasks, with the property that simply rolling it "
        "forward at inference time produces an agent that *learns* in-context. "
        "Algorithm Distillation (Laskin 2022) shows that distilling the source "
        "RL algorithm's learning curve into a transformer recovers a same-shape "
        "learning curve when the model is rolled forward on a new task. DPT "
        "(Lee 2023) takes this further with formal posterior-sampling guarantees. "
        "For matching, this matters because each recruiter / broker effectively "
        "runs their own task with a small amount of within-task data; an in-context "
        "policy can adapt without retraining."
    )
