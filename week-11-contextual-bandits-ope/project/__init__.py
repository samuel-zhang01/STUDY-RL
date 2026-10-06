"""Mini-project for week 11 — Contextual Bandits and Off-Policy Evaluation.

Builds a synthetic *synthetic insurance lead-broker* contextual bandit (6 brokers, 8 lead
features), trains LinUCB, LinTS and an epsilon-greedy logger, then runs SNIPS
and Doubly Robust off-policy evaluation against three counterfactual policies.

Target metric: LinTS achieves cumulative regret below 0.6 * T after T=2000
rounds on the default seed (T=2000, d=8, K=6). Runs in <20s on CPU.
"""

from . import env, agents, ope, train, solutions  # noqa: F401
