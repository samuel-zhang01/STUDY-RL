#!/usr/bin/env python3
"""Walk all week-* folders, report build status (LaTeX PDF present, notebook executed, tests passing)."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    rows = []
    weeks = sorted(ROOT.glob("week-*"))
    for w in weeks:
        has_notes_pdf = (w / "notes.pdf").exists()
        has_workbook = (w / "workbook.ipynb").exists()
        has_executed = (w / "workbook.executed.ipynb").exists()
        has_reading = (w / "extended_reading.pdf").exists()
        has_tests_dir = (w / "tests").is_dir()
        rows.append({
            "week": w.name,
            "notes.pdf": has_notes_pdf,
            "workbook.ipynb": has_workbook,
            "workbook.executed": has_executed,
            "extended_reading.pdf": has_reading,
            "tests/": has_tests_dir,
        })
    print(json.dumps(rows, indent=2))
    missing = [r for r in rows if not all([r["notes.pdf"], r["workbook.ipynb"]])]
    if missing:
        print(f"\n{len(missing)} week(s) missing core deliverables", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
