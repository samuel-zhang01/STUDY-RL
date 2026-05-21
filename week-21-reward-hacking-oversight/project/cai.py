"""Constitutional AI mini-loop.

Five-step pipeline:

1. **SFT.**  Train a tiny LM to produce English-like completions to
   ``"say: "``.
2. **Generate.**  Sample two candidates per prompt from the SFT model.
3. **Critique + revise.**  Apply a rule-based critique against a five-point
   constitution; if the candidate violates a principle, we *revise* it by
   sampling again (using a constitution-aware prompt). In a real CAI run
   the critique and revision would be done by a strong LLM; here we use a
   deterministic constitution function so the loop is testable.
4. **Preference pair.**  Score the original vs revised candidate by total
   constitution compliance; the higher-scoring one is the ``chosen``, the
   lower-scoring one is the ``rejected``.
5. **DPO step.**  One DPO gradient update on the synthetic preference pair.

Source: Bai et al., Constitutional AI, arXiv:2212.08073 (Anthropic, 2022).
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .agent import (
    CharLM, CharLMConfig, build_tokenizer, decode, encode,
    completion_mask, token_logprobs,
)
from .hack_lab import REF_CORPUS, _sft_warmup, HackConfig


# ----------------------------------------------------------------------------
# The constitution.
# ----------------------------------------------------------------------------


CONSTITUTION: tuple[str, ...] = (
    "Responses should contain no swearing or harsh language.",
    "Responses should not exceed 40 characters.",
    "Responses should be lowercase only.",
    "Responses should end with a period.",
    "Responses should avoid the character 'x'.",
)


# A *deterministic* compliance function. In real CAI, this would be a
# strong LLM judging against the constitution; we use a rule so the test
# suite is hermetic.
_BAD_WORDS = ("damn", "hell")


def critique(text: str) -> dict[str, bool]:
    """Return per-principle pass/fail booleans (True = pass)."""
    return {
        "no_swearing":      all(b not in text for b in _BAD_WORDS),
        "under_40_chars":   len(text) <= 40,
        "lowercase":        text == text.lower(),
        "ends_with_period": text.endswith("."),
        "no_x":             "x" not in text.lower(),
    }


def compliance_score(text: str) -> float:
    """Number of principles satisfied (0--5)."""
    return float(sum(critique(text).values()))


def revise(text: str) -> str:
    """Mechanical, rule-based revision applying each principle.

    Real CAI would use the policy itself to revise via a "rewrite to
    satisfy <constitution>" prompt; here we apply each principle
    deterministically so the resulting preference pair is reliable.
    """
    t = text
    for b in _BAD_WORDS:
        t = t.replace(b, "")
    t = t.lower()
    t = t.replace("x", "")  # after lower(), this catches both x and X

    if len(t) > 40:
        t = t[:40]
    if not t.endswith("."):
        t = t.rstrip() + "."
    return t


# ----------------------------------------------------------------------------
# DPO objective.
# ----------------------------------------------------------------------------


def dpo_loss(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """The DPO loss (Rafailov et al., 2023).

    .. math::
        \\mathcal{L}_{\\text{DPO}} = -\\E\\big[\\log \\sigma\\big( \\beta\\,
        (\\log\\pi_\\theta(y_w \\mid x) - \\log\\pi_\\theta(y_l \\mid x))
        - \\beta\\,(\\log\\pi_\\text{ref}(y_w \\mid x) - \\log\\pi_\\text{ref}(y_l \\mid x)) \\big)\\big].
    """
    pi_logits = beta * ((policy_logp_chosen - policy_logp_rejected)
                        - (ref_logp_chosen - ref_logp_rejected))
    return -F.logsigmoid(pi_logits).mean()


# ----------------------------------------------------------------------------
# Full CAI mini-loop.
# ----------------------------------------------------------------------------


@dataclass
class CAIConfig:
    num_steps: int = 100
    batch: int = 8
    max_new_tokens: int = 50
    temperature: float = 1.0
    learning_rate: float = 5e-3
    grad_clip: float = 1.0
    seed: int = 0
    beta: float = 0.1
    sft_warmup_steps: int = 200
    sft_lr: float = 5e-3


def _seqs_to_padded(
    seqs: list[list[int]], pad_id: int, device: torch.device,
) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs],
                        dtype=torch.long, device=device)


def cai_train(seed: int = 0, **kwargs) -> dict:
    """End-to-end CAI loop. Returns model + diagnostic history."""
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = CAIConfig(seed=seed, **kwargs)
    device = torch.device("cpu")
    stoi, itos = build_tokenizer()
    model = CharLM(CharLMConfig(vocab_size=len(stoi))).to(device)
    # Reuse the hack_lab SFT routine.
    _sft_warmup(model, stoi,
                HackConfig(sft_warmup_steps=cfg.sft_warmup_steps,
                           batch=cfg.batch, sft_lr=cfg.sft_lr, seed=cfg.seed),
                device)
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    bos, eos, pad = stoi["<bos>"], stoi["<eos>"], stoi["<pad>"]
    prompt = torch.tensor([[bos] + encode("say: ", stoi)],
                          dtype=torch.long, device=device)

    hist = {"step": [], "loss": [], "compliance_pre": [], "compliance_post": []}

    for step in range(cfg.num_steps):
        rep_prompt = prompt.expand(cfg.batch, -1).contiguous()
        L_p = rep_prompt.shape[1]

        # 1. Sample a candidate per prompt.
        model.eval()
        full_orig, _ = model.generate(rep_prompt, max_new_tokens=cfg.max_new_tokens,
                                       temperature=cfg.temperature, eos_id=eos)
        model.train()
        orig_texts = [decode(full_orig[i, L_p:].tolist(), itos) for i in range(cfg.batch)]
        # 2. Critique and revise -> build (chosen, rejected) pairs.
        chosen_seqs, rejected_seqs = [], []
        comp_pre, comp_post = [], []
        for orig in orig_texts:
            rev = revise(orig)
            c_orig = compliance_score(orig)
            c_rev = compliance_score(rev)
            comp_pre.append(c_orig); comp_post.append(c_rev)
            chosen_text = rev if c_rev >= c_orig else orig
            rejected_text = orig if c_rev >= c_orig else rev
            chosen_seqs.append(
                [bos] + encode("say: ", stoi) + encode(chosen_text, stoi) + [eos])
            rejected_seqs.append(
                [bos] + encode("say: ", stoi) + encode(rejected_text, stoi) + [eos])

        chosen_ids = _seqs_to_padded(chosen_seqs, pad, device)
        rejected_ids = _seqs_to_padded(rejected_seqs, pad, device)

        # 3. Sum log-probs over the completion span of each pair.
        def _seq_logp(model_, ids):
            logits = model_(ids[:, :-1])
            lp = F.log_softmax(logits, dim=-1)
            tgt = ids[:, 1:]
            chosen_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            comp_mask = (tgt != pad).float()
            # mask out the prompt prefix: positions corresponding to the
            # completion start at index L_p - 1 in the chosen_lp tensor.
            mask = comp_mask.clone()
            mask[:, :L_p - 1] = 0.0
            return (chosen_lp * mask).sum(dim=-1)

        policy_chosen_lp = _seq_logp(model, chosen_ids)
        policy_rejected_lp = _seq_logp(model, rejected_ids)
        with torch.no_grad():
            ref_chosen_lp = _seq_logp(ref_model, chosen_ids)
            ref_rejected_lp = _seq_logp(ref_model, rejected_ids)
        loss = dpo_loss(policy_chosen_lp, policy_rejected_lp,
                        ref_chosen_lp, ref_rejected_lp, beta=cfg.beta)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        hist["step"].append(step)
        hist["loss"].append(float(loss.item()))
        hist["compliance_pre"].append(float(np.mean(comp_pre)))
        hist["compliance_post"].append(float(np.mean(comp_post)))

    return {"model": model, "history": hist, "config": cfg, "stoi": stoi, "itos": itos}
