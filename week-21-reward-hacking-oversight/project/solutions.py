"""Reference solutions for the week-21 workbook."""
from __future__ import annotations
import numpy as np

from .cai import compliance_score, critique, revise, dpo_loss as _dpo_loss
from .hack_lab import proxy_reward, true_quality, KEYWORDS


# ---------------------------------------------------------------------------
# Exercise 1 ($\bigstar$). Goodhart classification.
# ---------------------------------------------------------------------------

def exercise_1(scenario: str) -> str:
    """Return one of ``{"regressional", "extremal", "causal", "adversarial"}``
    for a short scenario string. Mapping defined by the curriculum.
    """
    mapping = {
        "noise_overfitting":       "regressional",
        "out_of_distribution":     "extremal",
        "spurious_correlation":    "causal",
        "active_gaming":           "adversarial",
    }
    if scenario not in mapping:
        raise ValueError(f"unknown scenario: {scenario}")
    return mapping[scenario]


# ---------------------------------------------------------------------------
# Exercise 2 ($\bigstar\bigstar$). Implement the gameable proxy and the
# true-quality oracle disagreement coefficient.
# ---------------------------------------------------------------------------

def exercise_2(texts):
    """For a list of texts, return (proxy_scores, true_scores) and the
    *Spearman* rank correlation between them. A failing reward proxy will
    have correlation near zero or negative.
    """
    import scipy.stats as ss  # noqa: F401  -- imported lazily

    from .hack_lab import fit_bigram
    from .agent import build_tokenizer
    stoi, _ = build_tokenizer()
    log_bigram = fit_bigram(stoi)
    px = np.array([proxy_reward(t) for t in texts])
    tx = np.array([true_quality(t, log_bigram, stoi) for t in texts])
    # Hand-rolled Spearman to avoid the scipy dependency in tests.
    rx = np.argsort(np.argsort(px))
    rt = np.argsort(np.argsort(tx))
    rx, rt = rx - rx.mean(), rt - rt.mean()
    corr = float((rx * rt).sum() / (np.sqrt((rx**2).sum() * (rt**2).sum()) + 1e-12))
    return px, tx, corr


# ---------------------------------------------------------------------------
# Exercise 3 ($\bigstar\bigstar$). KL penalty schedule.
# ---------------------------------------------------------------------------

def exercise_3(step: int, beta0: float = 0.04, decay: float = 0.001) -> float:
    """Linear-decay KL coefficient schedule. Many RLHF recipes anneal beta
    from a strict value at the start (when the policy is closest to the
    reference) toward a looser value once training has progressed.
    """
    return max(beta0 - decay * step, 0.0)


# ---------------------------------------------------------------------------
# Exercise 4 ($\bigstar\bigstar$). Length penalty.
# ---------------------------------------------------------------------------

def exercise_4(reward: float, length: int, lam: float = 0.01) -> float:
    """Subtract a linear length penalty from the reward."""
    return reward - lam * length


# ---------------------------------------------------------------------------
# Exercise 5 ($\bigstar\bigstar$). Constitutional critique + revise.
# ---------------------------------------------------------------------------

def exercise_5(text: str) -> tuple[dict[str, bool], str, float, float]:
    """Returns the per-principle critique, the revised text, and the
    compliance scores before / after revision.
    """
    crit = critique(text)
    rev = revise(text)
    return crit, rev, compliance_score(text), compliance_score(rev)


# ---------------------------------------------------------------------------
# Exercise 6 ($\bigstar\bigstar$). DPO loss on a tiny batch.
# ---------------------------------------------------------------------------

def exercise_6(
    pol_c, pol_r, ref_c, ref_r, beta: float = 0.1,
):
    """Pure-NumPy DPO loss for verification against the torch version."""
    pi_logits = beta * ((pol_c - pol_r) - (ref_c - ref_r))
    # log sigmoid via softplus(-x).
    return float(np.log1p(np.exp(-pi_logits)).mean())


# ---------------------------------------------------------------------------
# Exercise 7 ($\bigstar\bigstar$). Debate equilibrium toy.
# ---------------------------------------------------------------------------

def exercise_7(p_judge_correct: float, n_rounds: int) -> float:
    """Probability the truthful side wins in a zero-sum debate where the
    judge has per-round accuracy ``p`` and the truthful side argues the
    correct claim in every round.

    Under the Irving-Christiano-Amodei 2018 model, with an oracle judge
    (p=1) the truthful side wins almost surely. As p degrades, the
    probability degrades; we model it as ``1 - (1-p)**n``: each round is
    an independent chance for the judge to detect the lie.
    """
    return 1.0 - (1.0 - p_judge_correct) ** n_rounds


# ---------------------------------------------------------------------------
# Exercise 8 ($\bigstar\bigstar\bigstar$, open-ended). Hack-vs-quality
# divergence experiment.
# ---------------------------------------------------------------------------

def exercise_8(history: dict, window: int = 20) -> float:
    """Return the *divergence index*: the gap between final-window mean
    proxy and final-window mean true quality, normalised by their initial
    gap. A value >1 indicates the hack has worsened the proxy-truth gap
    over training.
    """
    proxy = np.asarray(history["proxy"])
    truth = np.asarray(history["true_quality"])
    init = proxy[:window].mean() - truth[:window].mean()
    final = proxy[-window:].mean() - truth[-window:].mean()
    return float(final / (init + 1e-9))


__all__ = [
    "exercise_1", "exercise_2", "exercise_3", "exercise_4",
    "exercise_5", "exercise_6", "exercise_7", "exercise_8",
]
