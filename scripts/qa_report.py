#!/usr/bin/env python3
"""Report artifact presence across week-* folders, without executing any checks.

An executed-notebook filename or tests directory is not evidence that execution
passed. Use the build and test targets for fresh validation.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    rows = []
    weeks = sorted(w for w in ROOT.glob("week-*") if w.is_dir())
    if not weeks:
        print("No week bundles found", file=sys.stderr)
        return 1
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
            "teaching_guide.md": (w / "teaching_guide.md").is_file(),
            "project/solutions.py": (w / "project" / "solutions.py").is_file(),
            "notes.tex": (w / "notes.tex").is_file(),
            "extended_reading.tex": (w / "extended_reading.tex").is_file(),
        })
    print(json.dumps(rows, indent=2))
    # Executed notebooks are optional in this inventory: make notebooks supplies
    # fresh execution evidence. Require the rest of each shipped bundle.
    required = ("notes.pdf", "workbook.ipynb", "extended_reading.pdf", "tests/",
                "teaching_guide.md", "project/solutions.py", "notes.tex", "extended_reading.tex")
    missing = [r for r in rows if not all(r[key] for key in required)]
    if missing:
        print(f"\n{len(missing)} week(s) missing core deliverables", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
