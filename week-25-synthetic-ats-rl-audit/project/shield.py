"""Tiny shield: refuses to emit a ranking that violates a hard policy.

Hard policies in a real ATS include:
  * never surface a candidate flagged as duplicate / blacklisted
  * never surface a candidate without a valid right-to-work where the job
    requires one
  * never re-rank below a manually-pinned candidate

Here we implement a minimal example: forbid orderings where any candidate
in the "blocked" list appears in the top-k.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np


class HardConstraintViolation(Exception):
    pass


def shielded_rerank(order: np.ndarray, blocked: Iterable[int], top_k: int) -> np.ndarray:
    """Drop blocked indices from the top-k slice; raise if nothing survives."""
    blocked = set(int(b) for b in blocked)
    kept = [int(i) for i in order if int(i) not in blocked]
    if len(kept) < top_k:
        raise HardConstraintViolation(
            f"Only {len(kept)} candidates survive shield; need {top_k}."
        )
    return np.asarray(kept[:top_k], dtype=np.int64)
