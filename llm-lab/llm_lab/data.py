"""Reproducible, deliberately small addition data and a strict verifier.

Run from llm-lab: python -m llm_lab.data --output data
The arithmetic task makes preferences auditable; it is not a reasoning benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def score_completion(completion: str, expected: str) -> dict[str, bool]:
    """Accept one ASCII integer, allowing surrounding whitespace only.

    Do not execute generated code or reward a correct number hidden in prose.
    Format and correctness are separate so '42' cannot win every prompt.
    """
    text = completion.strip()
    valid = re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text) is not None
    return {"valid_format": valid, "exact_match": valid and text == expected.strip()}


def build_splits(seed: int = 42) -> dict[str, list[dict]]:
    """Group (a,b) and (b,a) before splitting to avoid trivial leakage."""
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(25) for b in range(a, 25)]
    rng.shuffle(pairs)
    templates = [
        "Compute {a} + {b}. Reply with just the integer.",
        "What is {a} plus {b}? Answer with only the integer.",
        "Add {a} and {b}. Return only the integer result.",
    ]
    result: dict[str, list[dict]] = {}
    start = 0
    for split, count in (("train", 192), ("validation", 32), ("test", 32)):
        rows = []
        for index, (a, b) in enumerate(pairs[start:start + count]):
            left, right = (b, a) if rng.random() < 0.5 else (a, b)
            rows.append({
                "id": f"add-{a:02d}-{b:02d}",
                "prompt": templates[index % len(templates)].format(a=left, b=right),
                "completion": str(a + b),
                "operands": [left, right],
            })
        result[f"sft_{split}"] = rows
        if split != "test":
            # Same training/validation groups across stages; no test preferences.
            result[f"preferences_{split}"] = [
                {"id": row["id"], "prompt": row["prompt"],
                 "chosen": row["completion"],
                 "rejected": str(int(row["completion"]) + (1 if i % 2 else -1))}
                for i, row in enumerate(rows)
            ]
        start += count
    return result


def write_splits(output: Path, seed: int = 42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator": "llm_lab.data.build_splits", "seed": seed,
        "provenance": "Synthetic integer addition, generated locally; no personal data.",
        "split_unit": "unordered operand pair", "files": {},
        "limitation": "Held out from this lab's training only; pretrained-model exposure is unknown.",
    }
    for name, rows in build_splits(seed).items():
        path = output / f"{name}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path.write_text(payload, encoding="utf-8")
        manifest["files"][path.name] = {
            "rows": len(rows), "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(write_splits(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
