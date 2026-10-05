"""The reward-hack lab.

The pedagogical setup:

* A tiny char-level LM is prompted with the fixed string ``"say: "``.
* A *deliberately gameable* proxy reward gives the policy credit for
  long completions that contain certain keywords.
* A *true quality* metric (hidden from the policy) measures how
  *coherent* the completion is, scored by a fixed bigram language model
  fit to a small corpus of English sentences.
* Without regularisation the policy maximises the proxy by spamming
  keywords and inflating length, while the true quality collapses --
  the central observation of Gao et al. 2022 and the formal definition
  of reward hacking in Skalse et al. 2024.
* Three mitigations are studied: a KL-to-reference anchor, an explicit
  length penalty, and an entropy bonus.

The whole pipeline runs in <30 s on CPU.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .agent import (
    CharLM, CharLMConfig, build_tokenizer, decode, encode,
    completion_mask, token_logprobs,
)


# ----------------------------------------------------------------------------
# Bigram language model for the *true* quality metric.
# ----------------------------------------------------------------------------


REF_CORPUS = [
    "the cat sat on the mat",
    "a dog ran across the field",
    "the sun rose over the hills",
    "she opened the door slowly",
    "rain fell on the quiet street",
    "the boy read a small book",
    "they walked along the river bank",
    "a soft wind moved the leaves",
    "the old man smiled at the child",
    "the moon shone over the lake",
    "music drifted from the open window",
    "stars filled the dark night sky",
    "the bird flew above the trees",
    "warm bread came from the oven",
    "snow covered the hills last night",
    "a small fire burned in the corner",
    "the child laughed at the story",
    "she wrote a long letter home",
    "the train arrived at the station",
    "leaves fell from the autumn trees",
]


def fit_bigram(stoi: dict[str, int], smoothing: float = 1.0) -> np.ndarray:
    """Add-1 smoothed bigram log-probabilities over the corpus."""
    V = len(stoi)
    counts = np.full((V, V), smoothing, dtype=np.float64)
    for sent in REF_CORPUS:
        ids = [stoi["<bos>"]] + encode(sent, stoi) + [stoi["<eos>"]]
        for a, b in zip(ids[:-1], ids[1:]):
            counts[a, b] += 1.0
    row_sums = counts.sum(axis=-1, keepdims=True)
    return np.log(counts / row_sums)


def true_quality(text: str, log_bigram: np.ndarray, stoi: dict[str, int]) -> float:
    """Per-character mean log-probability under the bigram LM. Higher is
    more "English-like"; near-zero / very negative is gibberish.

    Returns a score in roughly [-3, 0] for char-level English text.
    """
    ids = [stoi["<bos>"]] + encode(text, stoi) + [stoi["<eos>"]]
    if len(ids) < 2:
        return -10.0
    logp = 0.0
    for a, b in zip(ids[:-1], ids[1:]):
        logp += float(log_bigram[a, b])
    return logp / (len(ids) - 1)


# ----------------------------------------------------------------------------
# Proxy (gameable) reward.
# ----------------------------------------------------------------------------


KEYWORDS = ("good", "great", "best", "excellent")


def proxy_reward(text: str, alpha: float = 0.02, beta: float = 0.8) -> float:
    """Gameable proxy: ``alpha * length + beta * keyword_count``.

    The length term incentivises blathering, the keyword term incentivises
    spamming positive-sounding words. Designed to *correlate weakly* with
    true quality (a well-written sentence about something good will score
    a bit) but to be dominated by trivial gaming.
    """
    n_kw = sum(text.count(k) for k in KEYWORDS)
    return alpha * len(text) + beta * n_kw


# ----------------------------------------------------------------------------
# Training step (REINFORCE with optional KL / length / entropy regularisers).
# ----------------------------------------------------------------------------


@dataclass
class HackConfig:
    num_steps: int = 200
    batch: int = 16
    max_new_tokens: int = 40
    temperature: float = 1.0
    learning_rate: float = 5e-3
    grad_clip: float = 1.0
    seed: int = 0
    # Regularisers.
    kl_coef: float = 0.0
    length_penalty: float = 0.0   # subtract this * len(completion) from reward
    entropy_coef: float = 0.0
    # SFT warmup so the policy starts in "English-like" space.
    sft_warmup_steps: int = 200
    sft_lr: float = 5e-3
    # Reward shaping.
    proxy_alpha: float = 0.02
    proxy_beta: float = 0.8


def _sft_warmup(model, stoi, cfg, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.sft_lr)
    bos, eos, pad = stoi["<bos>"], stoi["<eos>"], stoi["<pad>"]
    rng = np.random.default_rng(cfg.seed)
    for _ in range(cfg.sft_warmup_steps):
        sents = rng.choice(REF_CORPUS, size=cfg.batch)
        seqs = [[bos] + encode(s, stoi) + [eos] for s in sents]
        L = max(len(s) for s in seqs)
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs],
                           dtype=torch.long, device=device)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1), ignore_index=pad)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()


def hack_train(seed: int = 0, **kwargs) -> dict:
    """Train the policy on the proxy reward with the given regularisation
    knobs. Records both proxy and true-quality scores over time so the
    hack-vs-quality divergence can be visualised.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = HackConfig(seed=seed, **kwargs)
    device = torch.device("cpu")  # tiny model; CPU is faster than CUDA setup.

    stoi, itos = build_tokenizer()
    log_bigram = fit_bigram(stoi)
    model = CharLM(CharLMConfig(vocab_size=len(stoi))).to(device)
    _sft_warmup(model, stoi, cfg, device)
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    bos = stoi["<bos>"]
    eos = stoi["<eos>"]
    prompt_ids_single = torch.tensor([[bos] + encode("say: ", stoi)],
                                     dtype=torch.long, device=device)

    hist = {
        "step": [], "proxy": [], "true_quality": [], "length": [],
        "keyword_rate": [], "kl": [], "entropy": [],
    }
    for step in range(cfg.num_steps):
        prompt_ids = prompt_ids_single.expand(cfg.batch, -1).contiguous()
        L_p = prompt_ids.shape[1]
        model.eval()
        full_ids, sample_logps = model.generate(
            prompt_ids, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, eos_id=eos,
        )
        model.train()
        comp_ids = full_ids[:, L_p:]
        mask = completion_mask(comp_ids, eos)
        texts = [decode(comp_ids[i].tolist(), itos) for i in range(cfg.batch)]
        proxies = np.array([proxy_reward(t, cfg.proxy_alpha, cfg.proxy_beta) for t in texts])
        truths = np.array([true_quality(t, log_bigram, stoi) for t in texts])
        lens = np.array([len(t) for t in texts])
        n_kw = np.array([sum(t.count(k) for k in KEYWORDS) for t in texts])

        # Effective reward = proxy minus optional length penalty.
        eff_reward = proxies - cfg.length_penalty * lens
        # Group baseline = mean over batch.
        baseline = (eff_reward.sum() - eff_reward) / max(1, len(eff_reward) - 1)
        adv = torch.tensor(eff_reward - baseline, dtype=torch.float32, device=device)

        logits = model(full_ids[:, :-1])[:, L_p - 1:] / cfg.temperature
        distribution = torch.distributions.Categorical(logits=logits)
        new_lp = distribution.log_prob(comp_ids)
        with torch.no_grad():
            ref_lp = token_logprobs(ref_model, full_ids, L_p, cfg.temperature)

        # KL (k3) and entropy on the sampling distribution.
        log_ratio = ref_lp - new_lp
        kl_tok = (torch.exp(log_ratio) - log_ratio - 1.0) * mask
        # Exact conditional entropy has a useful gradient. Treating sampled
        # -log pi tokens as a differentiable entropy loss alone omits the
        # score-function term and gives zero expected gradient.
        ent_tok = distribution.entropy() * mask

        # Plain REINFORCE on sequence-level reward, broadcast to tokens.
        pg_tok = -(new_lp * adv.unsqueeze(-1)) * mask
        denom = mask.sum(dim=-1).clamp_min(1.0)
        pg_loss = pg_tok.sum(dim=-1).mean()
        kl_loss = (kl_tok.sum(dim=-1) / denom).mean()
        ent_loss = -(ent_tok.sum(dim=-1) / denom).mean()  # negative -> maximise
        loss = pg_loss + cfg.kl_coef * kl_loss + cfg.entropy_coef * ent_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        hist["step"].append(step)
        hist["proxy"].append(float(proxies.mean()))
        hist["true_quality"].append(float(truths.mean()))
        hist["length"].append(float(lens.mean()))
        hist["keyword_rate"].append(float(n_kw.mean()))
        hist["kl"].append(float(kl_loss.item()))
        hist["entropy"].append(float(-ent_loss.item()))

    return {"model": model, "history": hist, "config": cfg,
            "log_bigram": log_bigram, "stoi": stoi, "itos": itos}


def evaluate_samples(result: dict, n: int = 8) -> list[str]:
    """Sample n completions from the trained policy --- useful for the
    workbook to inspect the qualitative degradation under reward hacking.
    """
    model = result["model"]
    cfg = result["config"]
    stoi, itos = result["stoi"], result["itos"]
    bos, eos = stoi["<bos>"], stoi["<eos>"]
    prompt = torch.tensor([[bos] + encode("say: ", stoi)] * n, dtype=torch.long)
    full, _ = model.generate(prompt, max_new_tokens=cfg.max_new_tokens,
                             temperature=cfg.temperature, eos_id=eos)
    L_p = prompt.shape[1]
    return [decode(full[i, L_p:].tolist(), itos) for i in range(n)]
