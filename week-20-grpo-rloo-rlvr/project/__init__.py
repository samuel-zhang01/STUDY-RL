"""Week 20 — GRPO / RLOO / RLVR with a toy arithmetic verifier.

Modules
-------
- ``env``      : the toy arithmetic prompt distribution and the verifier.
- ``agent``    : tiny character-level Transformer used as the policy.
- ``train``    : the GRPO / RLOO / Dr.GRPO training loops.
- ``solutions``: reference implementations of every workbook exercise.
"""
from .env import (
    ArithmeticTask,
    exact_match_reward,
    extract_final_number,
    sympy_equiv_reward,
)
from .agent import CharLM, build_tokenizer
from .train import train, grpo_step, rloo_step
from . import solutions

__all__ = [
    "ArithmeticTask",
    "exact_match_reward",
    "extract_final_number",
    "sympy_equiv_reward",
    "CharLM",
    "build_tokenizer",
    "train",
    "grpo_step",
    "rloo_step",
    "solutions",
]
