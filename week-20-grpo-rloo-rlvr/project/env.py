"""Toy arithmetic environment + verifiable-reward functions.

The "environment" generates prompts of the form ``"a+b="`` with one or two
digit operands and a single operator from ``{+, -}``. A completion is a string;
the verifier extracts the *last* integer in the completion and compares to the
ground truth.

This mirrors the structure of GSM8K / MATH RLVR pipelines used in DeepSeek-R1
and Tulu-3, but at a scale that fits on a CPU in <60s. The point is the
*signal*, not the difficulty.
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import Iterable

_NUM_RE = re.compile(r"-?\d+")


def extract_final_number(text: str) -> int | None:
    """Return the last signed integer in ``text``, or None if no number found.

    This mimics the answer-extraction step used by GSM8K-style RLVR (e.g. the
    ``####`` convention) but with a permissive regex so the policy can learn
    its own answer-marker.
    """
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def exact_match_reward(completion: str, target: int) -> float:
    """Binary 0/1 reward — the canonical RLVR signal.

    Hardened against the obvious hack of "no number found" returning a match.
    """
    pred = extract_final_number(completion)
    if pred is None:
        return 0.0
    return 1.0 if pred == target else 0.0


def sympy_equiv_reward(completion: str, target: int) -> float:
    """Slightly stronger verifier: allow algebraic equivalence.

    For our toy task this is overkill — exact-match suffices — but the API
    matches how production RLVR pipelines for MATH use ``sympy.simplify`` on
    extracted symbolic answers.
    """
    pred = extract_final_number(completion)
    if pred is None:
        return 0.0
    try:
        import sympy as sp
        return 1.0 if sp.simplify(pred - target) == 0 else 0.0
    except Exception:
        return 0.0


@dataclass
class ArithmeticTask:
    """A finite generator of (prompt, target) pairs.

    Parameters
    ----------
    max_operand: maximum operand magnitude (default 9 = single digit).
    ops:        which operators to allow.
    seed:       RNG seed.
    """

    max_operand: int = 9
    ops: tuple[str, ...] = ("+", "-")
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample(self) -> tuple[str, int]:
        a = self._rng.randint(0, self.max_operand)
        b = self._rng.randint(0, self.max_operand)
        op = self._rng.choice(self.ops)
        prompt = f"{a}{op}{b}="
        target = a + b if op == "+" else a - b
        return prompt, target

    def batch(self, n: int) -> list[tuple[str, int]]:
        return [self.sample() for _ in range(n)]

    def all_prompts(self) -> Iterable[tuple[str, int]]:
        """Enumerate every (a, op, b) — used to build a held-out test set."""
        for a in range(self.max_operand + 1):
            for b in range(self.max_operand + 1):
                for op in self.ops:
                    prompt = f"{a}{op}{b}="
                    target = a + b if op == "+" else a - b
                    yield prompt, target
