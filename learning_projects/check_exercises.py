"""Check learner work, or explicitly check references with --reference.

Pending exercises exit with status 1 and remain pending: no answer is substituted.
Use --exercise 1 to focus on one function, then progress through the five tasks.
"""
from __future__ import annotations

import argparse
import math
from typing import Callable

TASKS = {
    1: ("mean_update", [(1.0, (1, 1, 1)), (0.5, (1, 0, 2)), (2.0, (2, 2, 17)),
                         (-1.0, (1, -3, 2))]),
    2: ("discounted_return", [(3.6, ([0, 4], 0.9)), (2.75, ([2, 1, 1], 0.5)),
                               (0.0, ([], 0.9)), (7.0, ([2, 1, 4], 1.0)),
                               (2.0, ([2, 100], 0.0))]),
    3: ("q_target", [(2.6, (-1, 0.9, [4, -2], False)), (5.0, (5, 0.9, [100], True)),
                      (-3.0, (-3, 0.9, [], True)), (10.0, (1, 0.9, [10], False))]),
    4: ("ips", [(5 / 6, ([1, 0, 1, 1], [0.75, 0.75, 0.75, 0.25], [0.5] * 4)),
                   (4 / 3, ([1, 1], [0.75, 0.25], [0.5, 0.5])),
                   (0.0, ([1, 1], [1, 1], [0, 0]))]),
    5: ("shield", [(["safe"], (["safe", "risky"], {"safe": 0, "risky": 1}, 0)),
                     (["safe", "risky"], (["safe", "risky"], {"safe": 0, "risky": 1}, 1)),
                     ([], (["risky"], {"risky": 1}, 0))]),
}


def check(function: Callable, cases: list[tuple[object, tuple]]) -> tuple[str, str]:
    for expected, args in cases:
        try:
            actual = function(*args)
        except NotImplementedError as exc:
            return "PENDING", str(exc)
        except Exception as exc:
            return "FAIL", f"input {args!r} raised {type(exc).__name__}: {exc}"
        if isinstance(expected, list):
            matches = actual == expected
        else:
            try:
                matches = math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-9)
            except TypeError:
                matches = False
        if not matches:
            return "FAIL", f"input {args!r}: expected {expected!r}, got {actual!r}"
    return "PASS", f"{len(cases)} checks; explain why each boundary behaves this way"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true", help="explicitly validate reference answers")
    parser.add_argument("--exercise", type=int, choices=TASKS)
    args = parser.parse_args()
    from . import exercises, reference
    module = reference if args.reference else exercises
    print("Reference answers — this does not grade your implementation." if args.reference
          else "Your exercise functions — no reference fallback.")
    statuses = []
    for number, (name, cases) in TASKS.items():
        if args.exercise is not None and number != args.exercise:
            continue
        status, detail = check(getattr(module, name), cases)
        statuses.append(status)
        print(f"Exercise {number}: {name}: {status}: {detail}")
    if all(s == "PASS" for s in statuses):
        print("Now change one input, predict the result, and run your function again.")
        return 0
    print("Edit learning_projects/exercises.py; hints are in docs/INTERACTIVE_LEARNING.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
