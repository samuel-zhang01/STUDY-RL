"""Mini-project for week 02 — Bellman Equations and Dynamic Programming.

Builds on ``common.gridworld`` and ``common.tabular``. Adds:

- ``carrental.py``: Jack's car-rental MDP (a la Sutton & Barto, Example 4.2),
  in a slightly trimmed form so policy iteration finishes in a few seconds.
- ``solutions.py``: reference implementations of every workbook exercise.
- ``train.py``: drives value-iteration and policy-iteration to convergence on
  deterministic and stochastic GridWorlds + the car-rental MDP.
- ``make_figures.py``: deterministic figures used by ``notes.tex``.
"""
from .carrental import CarRentalMDP
