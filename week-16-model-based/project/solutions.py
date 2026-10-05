"""Reference solutions for Week 16 workbook."""
from __future__ import annotations
import numpy as np
import torch

from .world_model import RSSM, kl_gaussian
from .agent import DreamerLite
from .train import train, _lambda_return


def exercise_1_kl_diag_gaussian(mu_q, std_q, mu_p, std_p) -> torch.Tensor:
    """Closed-form KL(N(mu_q, std_q^2) || N(mu_p, std_p^2))."""
    return kl_gaussian(mu_q, std_q, mu_p, std_p)


def exercise_2_lambda_return(rewards: list, values: list, conts: list,
                              gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """Compute lambda-returns given lists of (B,) tensors of length H, H+1, H."""
    return _lambda_return(rewards, values, conts, gamma, lam)


def exercise_3_rssm_imagine(seed: int = 0) -> dict:
    """Build a tiny RSSM, imagine H steps from zero state, return final feature norm."""
    torch.manual_seed(seed)
    rssm = RSSM(obs_dim=4, act_dim=2, stoch_dim=8, det_dim=16, hidden=32)
    state = rssm.initial(2, torch.device("cpu"))
    a_oh = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    state, r, c = rssm.imagine_step(state, a_oh)
    return {"feat_norm": float(torch.cat([state.h, state.s], dim=-1).norm(dim=-1).mean()),
            "r": r.detach().numpy(),
            "c": c.detach().numpy()}


def exercise_4_train_dreamer(seed: int = 0, n_iters: int = 12):
    """Run a short Dreamer-lite training; return the eval curve."""
    return train(seed=seed, n_iters=n_iters)


def exercise_5_kl_balanced(mu_q, std_q, mu_p, std_p, alpha: float = 0.8) -> torch.Tensor:
    """KL-balancing trick (Dreamer V2): KL(stop(post)||prior) and KL(post||stop(prior))."""
    kl_lhs = kl_gaussian(mu_q.detach(), std_q.detach(), mu_p, std_p)
    kl_rhs = kl_gaussian(mu_q, std_q, mu_p.detach(), std_p.detach())
    return alpha * kl_lhs + (1 - alpha) * kl_rhs


def exercise_6_imagined_vs_real(out: dict, n_steps: int = 30) -> dict:
    """For each iteration, return whether returns are non-decreasing on average."""
    curve = out["eval_curve"]
    rets = curve[:, 1]
    return {"first": float(rets[0]), "last": float(rets[-1]),
            "improvement": float(rets[-1] - rets[0])}


def exercise_7_world_model_loss_components(seed: int = 0) -> dict:
    """Run two training iterations; return final mean world-model losses."""
    out = train(seed=seed, n_iters=2)
    return {**out["world_model_logs"][-1], "final_return": float(out["eval_curve"][-1, 1])}


def exercise_8_recruitment_world_model_demo() -> str:
    """Meaningful objective: explain why a world model is the right abstraction for recruitment."""
    return (
        "In recruitment matching the 'observation' is a partial candidate profile + "
        "stage history; the 'action' is the recruiter decision (advance / hold / "
        "reject); the 'reward' is the eventual outcome (offer / hire / no-show). A "
        "Dreamer-style world model encodes the latent 'candidate fit' state and "
        "lets the agent simulate forward what happens under hypothetical "
        "recruiter actions, without burning real candidates on bad decisions. "
        "Crucially, imagined rollouts are policy-improvement -- not policy-deployment."
    )
