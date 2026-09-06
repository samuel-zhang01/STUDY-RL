"""Generate held-out answers from a base model or a reloaded LoRA adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from . import DEFAULT_MODEL
from .data import score_completion
from .inference import generate_completion_details, load_inference_model


def evaluate_rows(model, tokenizer, rows: list[dict], max_new_tokens: int = 12) -> dict:
    if not rows:
        raise ValueError("Evaluation data must contain at least one row.")
    predictions = []
    start = time.perf_counter()
    for row in rows:
        details = generate_completion_details(model, tokenizer, row["prompt"], max_new_tokens, 0.0)
        scores = score_completion(details["text"], row["completion"])
        # A token-budget cutoff can hide a bad suffix. Only complete answers pass.
        scores = {name: passed and details["terminated"] for name, passed in scores.items()}
        predictions.append({"id": row.get("id"), "prompt": row["prompt"],
                            "expected": row["completion"], **details, **scores})
    n = len(predictions)
    correct = sum(row["exact_match"] for row in predictions)
    valid = sum(row["valid_format"] for row in predictions)
    return {
        "examples": n, "exact_matches": correct, "exact_match_rate": correct / n,
        "valid_formats": valid, "format_rate": valid / n,
        "length_cutoffs": sum(not row["terminated"] for row in predictions),
        "max_new_tokens": max_new_tokens, "decoding": "greedy",
        "generation_seconds": time.perf_counter() - start,
        "predictions": predictions,
        "interpretation": "Small lab-held-out addition set; not a general reasoning benchmark. Cutoff answers score zero.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--data", type=Path, default=Path("data/sft_test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--quantization", choices=["none", "nf4"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every row")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1:
        parser.error("limit must be nonnegative and max-new-tokens positive")
    if args.output.resolve() == args.data.resolve():
        parser.error("output must differ from the input dataset")
    try:
        payload = args.data.read_bytes()
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        for row in rows:
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
                raise ValueError("Evaluation rows need string prompt and completion fields.")
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            raise ValueError("Evaluation dataset is empty.")
        model, tokenizer, info = load_inference_model(
            args.model, args.adapter, args.device, args.quantization, args.revision, args.cpu_threads)
        report = evaluate_rows(model, tokenizer, rows, args.max_new_tokens)
        report["model"] = info
        report["data"] = {"path": str(args.data), "sha256": hashlib.sha256(payload).hexdigest()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
