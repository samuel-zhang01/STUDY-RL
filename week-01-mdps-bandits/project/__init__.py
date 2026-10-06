"""Mini-project for week 01 — Markov Decision Processes and Bandits.

Contents
--------
- ``env``: Bernoulli / Gaussian multi-armed bandit testbeds and a synthetic
  *synthetic insurance lead-broker* contextual bandit.
- ``agent``: epsilon-greedy, UCB1, Thompson Sampling, Boltzmann (softmax),
  LinUCB for the contextual setting.
- ``solutions``: reference implementations for every workbook exercise.
- ``train``: end-to-end experiment runner (converges in <60 s on CPU).
- ``make_figures``: deterministic figures used by ``notes.tex``.
"""
from .env import BernoulliBandit, GaussianBandit, SyntheticLeadBrokerBandit
from .agent import (
    EpsilonGreedyAgent,
    UCB1Agent,
    ThompsonBernoulliAgent,
    BoltzmannAgent,
    LinUCBAgent,
)

__all__ = [
    "BernoulliBandit",
    "GaussianBandit",
    "SyntheticLeadBrokerBandit",
    "EpsilonGreedyAgent",
    "UCB1Agent",
    "ThompsonBernoulliAgent",
    "BoltzmannAgent",
    "LinUCBAgent",
]
