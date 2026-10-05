"""Reject pytest collection that would mix the weeks' local package names.

Each bundle intentionally imports its own ``project`` and often ``tests``.
``make tests`` runs one bundle per process; default pytest covers shared tools.
"""
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    root = Path(__file__).resolve().parent
    weeks = set()
    mismatched = []
    for item in items:
        path = Path(item.path).resolve()
        try:
            first = path.relative_to(root).parts[0]
        except (ValueError, IndexError):
            continue
        if first.startswith("week-"):
            weeks.add(first)
        module = getattr(item, "module", None)
        source = getattr(module, "__file__", None)
        if source and Path(source).resolve() != path:
            mismatched.append(str(item.nodeid))
    if len(weeks) > 1 or mismatched:
        raise pytest.UsageError(
            "Weekly suites share local project/tests package names and must run "
            "in separate Python processes. Use `make tests` for the entire "
            "curriculum, or `python -m pytest week-NN-topic/tests` for one week. "
            "Combined collection can silently reuse another week's test module."
        )
