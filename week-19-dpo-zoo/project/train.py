"""Bake-off training driver for the DPO zoo."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple
import copy
import torch
import torch.nn.functional as F

from .data import ToyPreferenceDataset, encode, decode, VOCAB_SIZE
from .model import CharTransformer, TransformerConfig
from .sft import sft_train
from .losses import (
    dpo_loss, ipo_loss, kto_loss, orpo_loss, simpo_loss, cpo_loss,
    implicit_reward_margin,
)


def _pad(seqs, block):
    out = torch.zeros(len(seqs), block, dtype=torch.long)
    mask = torch.zeros(len(seqs), block, dtype=torch.bool)
    for i, s in enumerate(seqs):
        s = s[:block]
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return out, mask


def _seq_logprob(model: CharTransformer, ids: torch.Tensor, mask: torch.Tensor,
                  prompt_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (sum log p, per-token nll on response, response length)."""
    logits = model(ids[:, :-1])  # (B, T-1, V)
    targets = ids[:, 1:]
    logp_all = F.log_softmax(logits, dim=-1)
    gathered = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    # Only score positions strictly after the prompt.
    B, Tm1 = gathered.shape
    pos = torch.arange(Tm1, device=ids.device).unsqueeze(0).expand(B, Tm1)
    # Response mask: token at position p of the target corresponds to ids[:,p+1]
    # which is part of the response iff p+1 >= prompt_len.
    resp = (pos + 1 >= prompt_len.unsqueeze(1)) & mask[:, 1:]
    resp_f = resp.float()
    logp = (gathered * resp_f).sum(-1)
    length = resp_f.sum(-1)
    nll = -(gathered * resp_f).sum(-1) / length.clamp(min=1)
    return logp, nll, length


@dataclass
class BakeoffConfig:
    variant: str = "dpo"
    steps: int = 100
    lr: float = 1e-3
    batch_size: int = 8
    beta: float = 0.1
    gamma: float = 1.0  # SimPO target margin
    seed: int = 0


def train_variant(
    model: CharTransformer,
    ref: CharTransformer | None,
    pairs: Sequence[Tuple[str, str, str]],
    cfg: BakeoffConfig,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """Train ``model`` on the preference pairs under ``cfg.variant``.

    DPO/IPO need ``ref`` (frozen reference policy).
    SimPO/ORPO/CPO are reference-free; pass ``ref=None``.
    KTO uses ``ref`` plus a running KL estimate.
    """
    torch.manual_seed(cfg.seed)
    variant = cfg.variant.lower()
    if not pairs:
        raise ValueError("Preference training needs nonempty pairs")
    if variant in {"dpo", "ipo", "kto"} and ref is None:
        raise ValueError(f"{variant.upper()} requires a frozen reference model")
    model.to(device).train()
    if ref is not None:
        ref.to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    block = model.cfg.block_size

    enc_w = [encode(p + c) for (p, c, _) in pairs]
    enc_l = [encode(p + l) for (p, _, l) in pairs]
    plens = torch.tensor([len(encode(p)) for (p, _, _) in pairs], dtype=torch.long)
    if any(len(w) > block or len(l) > block for w, l in zip(enc_w, enc_l)):
        raise ValueError("Preference row exceeds block_size; shorten it rather than truncate the response")
    if any(len(w) <= p or len(l) <= p or p < 1 for w, l, p in zip(enc_w, enc_l, plens.tolist())):
        raise ValueError("Every pair needs an encodable prompt and two nonempty responses")

    history: Dict[str, List[float]] = {"loss": [], "margin": [], "acc": []}
    g = torch.Generator().manual_seed(cfg.seed)
    for step in range(cfg.steps):
        idx = torch.randint(0, len(pairs), (cfg.batch_size,), generator=g)
        w_ids, w_mask = _pad([enc_w[int(i)] for i in idx], block)
        l_ids, l_mask = _pad([enc_l[int(i)] for i in idx], block)
        pl = plens[idx]
        w_ids = w_ids.to(device); w_mask = w_mask.to(device)
        l_ids = l_ids.to(device); l_mask = l_mask.to(device); pl = pl.to(device)

        logp_w, nll_w, len_w = _seq_logprob(model, w_ids, w_mask, pl)
        logp_l, nll_l, len_l = _seq_logprob(model, l_ids, l_mask, pl)
        if ref is not None:
            with torch.no_grad():
                logp_ref_w, _, _ = _seq_logprob(ref, w_ids, w_mask, pl)
                logp_ref_l, _, _ = _seq_logprob(ref, l_ids, l_mask, pl)
        else:
            logp_ref_w = logp_ref_l = torch.zeros_like(logp_w)

        v = cfg.variant.lower()
        if v == "dpo":
            loss = dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta=cfg.beta)
        elif v == "ipo":
            loss = ipo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta=cfg.beta)
        elif v == "kto":
            # Alternate desirable/undesirable signals: winners desirable, losers undesirable
            # interleaved into the batch.
            ids = torch.cat([w_ids, l_ids], dim=0)
            masks = torch.cat([w_mask, l_mask], dim=0)
            pls = torch.cat([pl, pl], dim=0)
            logp, _, _ = _seq_logprob(model, ids, masks, pls)
            with torch.no_grad():
                logp_ref, _, _ = _seq_logprob(ref, ids, masks, pls)
            desirable = torch.cat([
                torch.ones(w_ids.size(0), device=device),
                torch.zeros(l_ids.size(0), device=device),
            ])
            # This toy uses an in-batch off-policy reference-point proxy. It
            # does not implement KTO's separate unpaired KL-estimation batch.
            kl_ref = (logp - logp_ref).detach().mean().clamp(min=0.0)
            loss = kto_loss(logp, logp_ref, desirable, kl_ref, beta=cfg.beta)
        elif v == "orpo":
            loss = orpo_loss(-nll_w, -nll_l, nll_w, beta=cfg.beta)
        elif v == "simpo":
            loss = simpo_loss(logp_w, logp_l, len_w, len_l, beta=cfg.beta, gamma=cfg.gamma)
        elif v == "cpo":
            loss = cpo_loss(logp_w, logp_l, nll_w, beta=cfg.beta)
        else:
            raise ValueError(f"unknown variant {cfg.variant}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            if v in ("dpo", "ipo", "kto"):
                margin = implicit_reward_margin(
                    logp_w, logp_l, logp_ref_w, logp_ref_l, beta=cfg.beta
                )
            elif v == "simpo":
                margin = cfg.beta * (logp_w / len_w.clamp(min=1) - logp_l / len_l.clamp(min=1))
            else:
                margin = cfg.beta * (logp_w - logp_l)
            history["loss"].append(float(loss.item()))
            history["margin"].append(float(margin.mean().item()))
            history["acc"].append(float((margin > 0).float().mean().item()))
    return history


def bakeoff(
    pairs: Sequence[Tuple[str, str, str]],
    sft_strings: Sequence[str],
    variants: Sequence[str] = ("dpo", "kto", "simpo"),
    sft_steps: int = 100,
    pref_steps: int = 80,
    seed: int = 0,
) -> Dict[str, Dict[str, List[float]]]:
    """Run SFT once, then train one copy per variant from that checkpoint."""
    torch.manual_seed(seed)
    base = CharTransformer()
    sft_train(base, sft_strings, steps=sft_steps, seed=seed)
    sft_state = copy.deepcopy(base.state_dict())

    results: Dict[str, Dict[str, List[float]]] = {}
    for v in variants:
        model = CharTransformer()
        model.load_state_dict(sft_state)
        ref = CharTransformer()
        ref.load_state_dict(sft_state)
        results[v] = train_variant(
            model, ref if v in ("dpo", "ipo", "kto") else None,
            pairs, BakeoffConfig(variant=v, steps=pref_steps, seed=seed),
        )
    return results


def train(seed: int = 0, **kwargs):
    """Default entry point.  Returns the bake-off dict."""
    ds = ToyPreferenceDataset(n=kwargs.get("n_pairs", 64), seed=seed)
    return bakeoff(
        ds.pairs, ds.sft_strings(),
        variants=kwargs.get("variants", ("dpo", "kto", "simpo")),
        sft_steps=kwargs.get("sft_steps", 100),
        pref_steps=kwargs.get("pref_steps", 80),
        seed=seed,
    )


if __name__ == "__main__":
    results = train()
    for v, h in results.items():
        print(f"{v}: final margin = {h['margin'][-1]:.3f}  acc = {h['acc'][-1]:.3f}")
