"""Tiny toy preference dataset.

We use a character-level vocabulary and a synthetic "style" preference
signal: prompts are short strings of the form ``"Q: <topic>?  A:"`` and
the *preferred* completions are short, polite, and end with a period;
the *rejected* completions are long, shouty (upper-case), or end with
``"!!"``.  This gives the reward model an unambiguous, learnable
signal in a few seconds, which is exactly what we need to demo PPO
without spending an hour on real data.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Tuple

ALPHABET = (
    " abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789.,?!:;'-"
)
PAD = "\x00"
VOCAB = PAD + ALPHABET
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(VOCAB)


def encode(s: str) -> List[int]:
    return [STOI.get(c, 0) for c in s]


def decode(ids) -> str:
    return "".join(ITOS.get(int(i), "?") for i in ids)


TOPICS = [
    "cats", "dogs", "rain", "code", "music", "books",
    "trees", "stars", "fish", "wind", "trains", "tea",
]
POLITE_TEMPLATES = [
    "yes, {t} are nice.",
    "i like {t} a lot.",
    "{t} can be lovely.",
    "they are pleasant.",
    "it depends, but yes.",
]
RUDE_TEMPLATES = [
    "{T} ARE TERRIBLE!!",
    "NO!! GO AWAY!!",
    "WHO CARES ABOUT {T}!!",
    "STUPID QUESTION!!",
    "I HATE {T}!!",
]


def _prompt(topic: str) -> str:
    return f"Q: do you like {topic}? A:"


@dataclass
class ToyPreferenceDataset:
    """Yields (prompt, chosen, rejected) triples."""

    n: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        pairs: List[Tuple[str, str, str]] = []
        for _ in range(self.n):
            t = rng.choice(TOPICS)
            prompt = _prompt(t)
            chosen = " " + rng.choice(POLITE_TEMPLATES).format(t=t)
            rejected = " " + rng.choice(RUDE_TEMPLATES).format(T=t.upper())
            pairs.append((prompt, chosen, rejected))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[str, str, str]:
        return self.pairs[i]

    def sft_strings(self) -> List[str]:
        """Concatenation of prompt + chosen for SFT."""
        return [p + c for (p, c, _) in self.pairs]

    def prompts(self) -> List[str]:
        return [p for (p, _, _) in self.pairs]


def heuristic_reward(text: str) -> float:
    """A hand-coded oracle reward used to *seed* the toy preference data
    and as a sanity check for the learned reward model.  We never give
    this oracle directly to PPO — only the learned RM is used."""
    # reward shortness, lower case, and a single full stop.
    body = text.split("A:")[-1]
    score = 0.0
    score += 1.0 if "." in body and "!!" not in body else 0.0
    score += 0.5 if body.strip().islower() or not any(c.isupper() for c in body) else -0.5
    score += -0.05 * max(0, len(body) - 25)  # length penalty
    return score
