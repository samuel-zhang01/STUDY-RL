"""Toy preference dataset for week 19 -- identical to week 18 to keep
the comparison fair.  See ``week-18-rlhf-ppo-llm/project/data.py`` for
commentary.
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
POLITE = [
    "yes, {t} are nice.",
    "i like {t} a lot.",
    "{t} can be lovely.",
    "they are pleasant.",
    "it depends, but yes.",
]
RUDE = [
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
    n: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        pairs: List[Tuple[str, str, str]] = []
        for _ in range(self.n):
            t = rng.choice(TOPICS)
            p = _prompt(t)
            c = " " + rng.choice(POLITE).format(t=t)
            r = " " + rng.choice(RUDE).format(T=t.upper())
            pairs.append((p, c, r))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        return self.pairs[i]

    def prompts(self) -> List[str]:
        return [p for (p, _, _) in self.pairs]

    def sft_strings(self) -> List[str]:
        return [p + c for (p, c, _) in self.pairs]
