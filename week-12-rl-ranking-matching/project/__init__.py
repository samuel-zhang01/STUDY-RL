"""Mini-project for week 12 — RL for Ranking and Two-Sided Matching.

The flagship applied lecture: two-tower retrieval (CV <-> JD) followed by a
LinTS contextual-bandit re-ranker, with CLTR, fairness audit, and
calibration. The synthetic dataset is bundled (200 candidates x 50 jobs);
no external downloads required.

Target metrics (default seed):
- Two-tower NDCG@10 > 0.5 on the ground-truth Jaccard relevance.
- LinTS reranker at least matches two-tower NDCG@10 (often exceeds).
- Calibration: Beta-calibration ECE < raw-score ECE.
Runtime: ~30s on CPU.
"""

from . import env, two_tower, reranker, cltr, fairness, calibration, train, solutions  # noqa: F401
