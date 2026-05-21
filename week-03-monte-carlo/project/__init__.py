"""Mini-project for week 03 — Monte Carlo methods.

Contents
--------
- ``blackjack``: a tabular Blackjack environment matching Sutton & Barto's
  formulation (Example 5.1).
- ``recruitment``: a synthetic *delayed-feedback recruitment funnel* MDP that
  exercises off-policy MC on an applied matching problem.
- ``solutions``: reference implementations for the workbook.
- ``train``: end-to-end driver (converges in <60 s on CPU).
- ``make_figures``: deterministic figures used by ``notes.tex``.
"""
from .blackjack import BlackjackEnv
from .recruitment import RecruitmentMDP
