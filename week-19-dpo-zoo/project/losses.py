"""The DPO zoo: six preference-optimisation losses, written out so
their differences are visible at a glance.

Notation throughout:

* ``logp_w``: log pi_theta(y_w | x)  -- summed over the response tokens.
* ``logp_l``: log pi_theta(y_l | x).
* ``logp_ref_w``, ``logp_ref_l``: same but under the frozen reference.
* ``len_w``, ``len_l``: response lengths in tokens (used by SimPO).
* ``beta``, ``gamma``: per-variant hyper-parameters.

Each function returns a *scalar* loss (per-batch mean).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. DPO -- Rafailov et al. 2023
# ---------------------------------------------------------------------------

def dpo_loss(
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    logp_ref_w: torch.Tensor,
    logp_ref_l: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """L_DPO = -E log sigmoid( beta * (lr_w - lr_l) )

    where lr_w = log pi_theta(y_w|x) - log pi_ref(y_w|x).
    """
    lr_w = logp_w - logp_ref_w
    lr_l = logp_l - logp_ref_l
    return -F.logsigmoid(beta * (lr_w - lr_l)).mean()


# ---------------------------------------------------------------------------
# 2. IPO -- Azar et al. 2024.  Squared-loss objective that does not
#    saturate under deterministic preferences.
# ---------------------------------------------------------------------------

def ipo_loss(
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    logp_ref_w: torch.Tensor,
    logp_ref_l: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """L_IPO = E ( (lr_w - lr_l) - 1/(2 beta) )^2

    With sigmoid removed; the target log-ratio gap is 1/(2 beta).
    """
    lr_w = logp_w - logp_ref_w
    lr_l = logp_l - logp_ref_l
    return ((lr_w - lr_l - 1.0 / (2.0 * beta)) ** 2).mean()


# ---------------------------------------------------------------------------
# 3. KTO -- Ethayarajh et al. 2024.  Unpaired Kahneman--Tversky utility.
# ---------------------------------------------------------------------------

def kto_loss(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
    is_desirable: torch.Tensor,
    kl_ref: torch.Tensor,
    beta: float = 0.1,
    lam_pos: float = 1.0,
    lam_neg: float = 1.0,
) -> torch.Tensor:
    """L_KTO = lam_pos * (1 - sigmoid( beta*(lr - kl_ref))) on desirable
       + lam_neg * (1 - sigmoid( beta*(kl_ref - lr))) on undesirable.

    ``kl_ref`` is an estimate of KL(pi_theta || pi_ref) on a *different*
    batch -- the "reference point" of prospect theory.
    """
    lr = logp - logp_ref  # log ratio
    pos = 1.0 - torch.sigmoid(beta * (lr - kl_ref))
    neg = 1.0 - torch.sigmoid(beta * (kl_ref - lr))
    w = is_desirable.float()
    return (lam_pos * w * pos + lam_neg * (1.0 - w) * neg).mean()


# ---------------------------------------------------------------------------
# 4. ORPO -- Hong et al. 2024.  Odds-ratio + SFT, no reference model.
# ---------------------------------------------------------------------------

def orpo_loss(
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    nll_w: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """L_ORPO = NLL(y_w | x)
              - beta * log sigmoid( log(odds(y_w)/odds(y_l)) )

    where odds(y) = pi_theta(y) / (1 - pi_theta(y)).
    In log space: log odds = logp - log(1 - exp(logp))
                           = logp - log1mexp(logp).
    """
    log_odds_w = logp_w - _log1mexp(logp_w)
    log_odds_l = logp_l - _log1mexp(logp_l)
    odds_ratio = log_odds_w - log_odds_l
    return nll_w.mean() - beta * F.logsigmoid(odds_ratio).mean()


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x <= 0."""
    # Mächler 2012.
    return torch.where(
        x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x))
    )


# ---------------------------------------------------------------------------
# 5. SimPO -- Meng, Xia, Chen 2024.  Reference-free, length-normalised.
# ---------------------------------------------------------------------------

def simpo_loss(
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    len_w: torch.Tensor,
    len_l: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """L_SimPO = - log sigmoid( beta * (logp_w/|y_w| - logp_l/|y_l|) - gamma )"""
    avg_w = logp_w / len_w.clamp(min=1).float()
    avg_l = logp_l / len_l.clamp(min=1).float()
    return -F.logsigmoid(beta * (avg_w - avg_l) - gamma).mean()


# ---------------------------------------------------------------------------
# 6. CPO -- Xu et al. 2024.  SFT NLL + contrastive log-ratio (no ref).
# ---------------------------------------------------------------------------

def cpo_loss(
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    nll_w: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """L_CPO = NLL(y_w | x) - log sigmoid( beta * (logp_w - logp_l) )

    Distinct from ORPO: a plain log-ratio (not odds-ratio), still no
    reference model, still combined with SFT NLL.
    """
    return nll_w.mean() - F.logsigmoid(beta * (logp_w - logp_l)).mean()


# ---------------------------------------------------------------------------
# Reward-margin diagnostic shared across variants.
# ---------------------------------------------------------------------------

def implicit_reward_margin(
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    logp_ref_w: torch.Tensor,
    logp_ref_l: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """beta * ((log pi - log pi_ref) on winner) - (same on loser).
    The expected DPO implicit reward gap."""
    return beta * ((logp_w - logp_ref_w) - (logp_l - logp_ref_l))
