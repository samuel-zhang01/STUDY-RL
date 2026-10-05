"""PPO for language models — stage 3 of RLHF.

A nanoGPT-sized, from-scratch implementation that mirrors the textbook
RLHF objective:

    J(theta) = E_{x, y ~ pi_theta} [ r_phi(x, y) - beta * KL(pi_theta || pi_ref) ]

with per-token credit assignment.  Key choices, all explicit:

* The *reward* is the scalar reward-model score at the final token of
  the completion; a *per-token KL penalty* is added to every step.
* Advantages are estimated with Generalized Advantage Estimation
  (GAE) on the token sequence.
* PPO clipped surrogate; a separate value head shares the trunk with
  the policy and is trained with MSE against returns.
* A proportional adaptive KL controller uses the sampled policy-to-frozen-
  reference log-ratio, averaged per generated token.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
import copy
import math
import torch
import torch.nn.functional as F

from .model import CharTransformer, RewardModel, ValueHead
from .data import encode, decode, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Generalised Advantage Estimation (token level)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,  # (B, T)
    values: torch.Tensor,   # (B, T+1) — bootstrap appended
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (advantages, returns), both (B, T)."""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    returns = adv + values[:, :T]
    return adv, returns


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    prompt_ids: torch.Tensor     # (B, P)
    response_ids: torch.Tensor   # (B, T)
    logprobs: torch.Tensor       # (B, T) — policy at sample time
    ref_logprobs: torch.Tensor   # (B, T) — reference policy
    values: torch.Tensor         # (B, T+1)
    rewards: torch.Tensor        # (B, T) — per-token (KL penalty + final r)
    rm_scores: torch.Tensor      # (B,)   — raw reward-model output
    temperature: float = 1.0    # same distribution in sampling and PPO update


def _gather_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-prob of `targets` under `logits` for each position."""
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def sample_rollouts(
    policy: CharTransformer,
    value_head: ValueHead,
    ref: CharTransformer,
    rm: RewardModel,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float = 1.0,
    kl_coef: float = 0.05,
    device: str = "cpu",
) -> Rollout:
    """Sample `len(prompts)` responses and assemble a token-level rollout
    with per-token KL-shaped rewards.
    """
    block = policy.cfg.block_size
    if not prompts or not 0 < max_new_tokens < block:
        raise ValueError("Need nonempty prompts and 0 < max_new_tokens < block_size")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    enc = [encode(p)[-block + max_new_tokens:] for p in prompts]
    if any(not e for e in enc):
        raise ValueError("Each prompt needs at least one encodable token")
    P = max(len(e) for e in enc)
    prompt_ids = torch.zeros(len(prompts), P, dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        prompt_ids[i, P - len(e):] = torch.tensor(e, dtype=torch.long)

    # Autoregressive sampling.
    idx = prompt_ids.clone()
    response_logits_pi: List[torch.Tensor] = []
    response_logits_ref: List[torch.Tensor] = []
    response_tokens: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    for t in range(max_new_tokens):
        ctx = idx[:, -block:]
        h_pi = policy.trunk(ctx)
        logits_pi = policy.lm_head(h_pi)[:, -1, :] / max(temperature, 1e-6)
        h_ref = ref.trunk(ctx)
        logits_ref = ref.lm_head(h_ref)[:, -1, :] / max(temperature, 1e-6)
        v_t = value_head(h_pi[:, -1, :].unsqueeze(1)).squeeze(1)
        probs = F.softmax(logits_pi, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        response_logits_pi.append(logits_pi)
        response_logits_ref.append(logits_ref)
        response_tokens.append(nxt)
        values.append(v_t)
    # Bootstrap value at terminal step (use 0 — end of episode).
    values.append(torch.zeros_like(values[-1]))

    response_ids = torch.cat(response_tokens, dim=1)  # (B, T)
    logits_pi = torch.stack(response_logits_pi, dim=1)  # (B, T, V)
    logits_ref = torch.stack(response_logits_ref, dim=1)
    logprobs = _gather_logprobs(logits_pi, response_ids)
    ref_logprobs = _gather_logprobs(logits_ref, response_ids)
    values_t = torch.stack(values, dim=1)  # (B, T+1)

    # Reward model score on the full sequence.
    full = torch.cat([prompt_ids, response_ids], dim=1)
    lengths = torch.full((full.size(0),), full.size(1), dtype=torch.long, device=device)
    rm_scores = rm(full[:, -block:], lengths.clamp(max=block))

    # Per-token KL penalty + sparse final reward.
    kl_token = logprobs - ref_logprobs  # log pi - log pi_ref
    per_tok_rewards = -kl_coef * kl_token
    per_tok_rewards[:, -1] = per_tok_rewards[:, -1] + rm_scores

    return Rollout(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        logprobs=logprobs.detach(),
        ref_logprobs=ref_logprobs.detach(),
        values=values_t.detach(),
        rewards=per_tok_rewards.detach(),
        rm_scores=rm_scores.detach(),
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_step(
    policy: CharTransformer,
    value_head: ValueHead,
    rollout: Rollout,
    optimizer: torch.optim.Optimizer,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    n_epochs: int = 4,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> dict:
    """One PPO update on a rollout.  Returns dict of diagnostics."""
    adv, returns = compute_gae(rollout.rewards, rollout.values, gamma, lam)
    # Normalise advantages.
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    block = policy.cfg.block_size
    full = torch.cat([rollout.prompt_ids, rollout.response_ids], dim=1)[:, -block:]
    P = rollout.prompt_ids.size(1)
    T = rollout.response_ids.size(1)
    # Build labels: positions corresponding to the response tokens.
    # We need to compute log pi_theta at the *prompt-conditioned* positions
    # for each response token.  Use one forward pass and slice.

    diagnostics = {}
    for epoch in range(n_epochs):
        h = policy.trunk(full)
        logits = policy.lm_head(h)
        # The token at position k is predicted from logits[k-1].
        # We want the log-prob of response_ids[t] from position (P + t - 1).
        # In the (possibly truncated) full sequence of length L,
        # the response starts at index L - T, so we slice logits[:, L-T-1:L-1].
        L = full.size(1)
        resp_logits = logits[:, L - T - 1: L - 1, :] / rollout.temperature
        new_logprobs = _gather_logprobs(resp_logits, rollout.response_ids)

        ratio = (new_logprobs - rollout.logprobs).exp()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        policy_loss = -torch.min(unclipped, clipped).mean()

        v_pred = value_head(h[:, L - T - 1: L - 1, :])
        value_loss = F.mse_loss(v_pred, returns)

        # Entropy bonus (optional).
        probs = F.softmax(resp_logits, dim=-1)
        logp = F.log_softmax(resp_logits, dim=-1)
        entropy = -(probs * logp).sum(-1).mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_head.parameters()), 1.0
        )
        optimizer.step()

        with torch.no_grad():
            approx_kl = (rollout.logprobs - new_logprobs).mean().item()
        diagnostics = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl),
            # Signed Monte Carlo estimate of KL(old policy || reference),
            # distinct from the PPO old/new update diagnostic above.
            "reference_kl": float((rollout.logprobs - rollout.ref_logprobs).mean()),
            "mean_reward": float(rollout.rm_scores.mean().item()),
        }
    return diagnostics


# ---------------------------------------------------------------------------
# Proportional adaptive KL controller
# ---------------------------------------------------------------------------

class AdaptiveKLController:
    """Update beta multiplicatively to keep observed KL near a target."""

    def __init__(self, init_beta: float = 0.05, target_kl: float = 0.1, horizon: int = 10):
        if init_beta < 0 or target_kl <= 0 or horizon < 1:
            raise ValueError("beta must be nonnegative; target and horizon positive")
        self.beta = init_beta
        self.target = target_kl
        self.horizon = horizon

    def update(self, observed_kl: float) -> None:
        # Cap proportional feedback; this heuristic is not a hard KL bound.
        proportional_error = max(-0.2, min(0.2, observed_kl / self.target - 1.0))
        self.beta *= 1.0 + proportional_error * (1.0 / self.horizon)


# ---------------------------------------------------------------------------
# Top-level loop
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    steps: int = 30
    batch_size: int = 8
    max_new_tokens: int = 16
    temperature: float = 1.0
    lr: float = 1e-3
    clip_eps: float = 0.2
    n_epochs: int = 2
    init_kl_coef: float = 0.05
    target_kl: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    seed: int = 0


def ppo_train(
    policy: CharTransformer,
    rm: RewardModel,
    prompts: Sequence[str],
    cfg: PPOConfig | None = None,
    device: str = "cpu",
):
    """End-to-end PPO loop on a fixed prompt set.  Returns history dict."""
    cfg = cfg or PPOConfig()
    torch.manual_seed(cfg.seed)
    policy.to(device).train()
    rm.to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    # Frozen reference = snapshot of the SFT policy.
    ref = copy.deepcopy(policy).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    value_head = ValueHead(policy.cfg.d_model).to(device)
    opt = torch.optim.AdamW(
        list(policy.parameters()) + list(value_head.parameters()), lr=cfg.lr
    )
    kl_ctl = AdaptiveKLController(cfg.init_kl_coef, cfg.target_kl)

    g = torch.Generator().manual_seed(cfg.seed)
    history = {"reward": [], "kl": [], "policy_loss": [], "value_loss": [], "beta": []}
    for step in range(cfg.steps):
        idx = torch.randint(0, len(prompts), (cfg.batch_size,), generator=g)
        batch = [prompts[int(i)] for i in idx]
        rollout = sample_rollouts(
            policy, value_head, ref, rm,
            prompts=batch,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            kl_coef=kl_ctl.beta,
            device=device,
        )
        diag = ppo_step(
            policy, value_head, rollout, opt,
            clip_eps=cfg.clip_eps, n_epochs=cfg.n_epochs,
            gamma=cfg.gamma, lam=cfg.lam,
        )
        kl_ctl.update(diag["reference_kl"])
        history["reward"].append(diag["mean_reward"])
        history["kl"].append(diag["reference_kl"])
        history["policy_loss"].append(diag["policy_loss"])
        history["value_loss"].append(diag["value_loss"])
        history["beta"].append(kl_ctl.beta)
    return history, value_head, ref


def train(seed: int = 0, **kwargs):
    """Convergence target: PPO reward curve trends upward against an SFT
    baseline in under 60 s on CPU.  Documented as such in tests.
    """
    from .data import ToyPreferenceDataset
    from .model import CharTransformer, RewardModel, TransformerConfig
    torch.manual_seed(seed)
    ds = ToyPreferenceDataset(n=64, seed=seed)
    cfg = TransformerConfig()
    policy = CharTransformer(cfg)
    rm_backbone = CharTransformer(cfg)
    rm = RewardModel(rm_backbone)
    # Stage 1: SFT
    from .sft import sft_train
    sft_train(policy, ds.sft_strings(), steps=kwargs.get("sft_steps", 150), seed=seed)
    # Stage 2: RM
    from .rm import rm_train
    rm_train(rm, ds.pairs, steps=kwargs.get("rm_steps", 150), seed=seed)
    # Stage 3: PPO
    pcfg = PPOConfig(
        steps=kwargs.get("ppo_steps", 30),
        batch_size=kwargs.get("batch_size", 4),
        max_new_tokens=kwargs.get("max_new_tokens", 16),
        seed=seed,
    )
    return ppo_train(policy, rm, ds.prompts(), pcfg)


if __name__ == "__main__":
    hist, _, _ = train()
    print("Final mean reward:", hist["reward"][-1])
