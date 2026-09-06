#!/usr/bin/env python3
"""Consolidate exercise-specific inline hints without revealing reference code.

Existing generated sections are preserved unless --refresh is supplied. Refresh
only replaces cells marked AUTO-HINTS-SECTION; learner cells are left in place.
The generated section is excluded from extraction so it cannot copy itself.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HINTS_MARKER = "<!-- AUTO-HINTS-SECTION -->"
EXERCISE_HEADING = re.compile(
    r"^\s*#{1,4}\s*((?:Exercise|Ex|E|Question|Q|Task|Problem)\s*\d+[^\n]*"
    r"|\d+\.\s+[A-Z][^\n]*)", re.IGNORECASE | re.MULTILINE,
)
HINT_START = re.compile(r"^\s*(?:>\s*)?(?:\*{1,2})?Hint\b|\(Hint\s*:", re.IGNORECASE)


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def find_exercises(nb: dict) -> list[tuple[str, str]]:
    exercises = []
    seen = set()
    # A prompt and its hint often occupy different Markdown cells. Join their
    # narrative text before locating exercise boundaries; never include the
    # generated section or any executable cell in this extraction.
    narrative = "\n\n".join(source(cell) for cell in nb["cells"]
                             if cell.get("cell_type") == "markdown"
                             and HINTS_MARKER not in source(cell))
    for text in [narrative]:
        headings = list(EXERCISE_HEADING.finditer(text))
        for i, match in enumerate(headings):
            title = match.group(1).strip().rstrip("—-:").strip()
            if title in seen:
                continue
            seen.add(title)
            stop = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            lines = text[match.end():stop].splitlines()
            hints = []
            collecting = False
            for line in lines:
                hint_start = HINT_START.search(line)
                if hint_start:
                    collecting = True
                    line = line[hint_start.start():]
                elif not line.strip() or line.lstrip().startswith(("#", "</details>")):
                    collecting = False
                if collecting:
                    hints.append(line)
            exercises.append((title, "\n".join(hints)))
    return exercises


def make_hints_cell(exercises: list[tuple[str, str]]) -> dict:
    body = [
        HINTS_MARKER + "\n",
        "## Hints toward answers\n\n",
        "Attempt the exercise first. Before opening a hint, write the quantity you are\n",
        "estimating, the data source, and the expected update direction. The\n",
        "[teaching guide](teaching_guide.md) supplies worked examples, assumptions,\n",
        "and transfer questions. The hints below collect the exercise's own inline\n",
        "hints; they do not reveal or execute reference solutions.\n\n",
    ]
    for title, hint in exercises:
        body.extend([
            "<details>\n",
            f"<summary>{html.escape(title)}</summary>\n\n",
            (hint if hint else "No inline hint is supplied for this exercise. Use the corresponding derivation and code bridge in the teaching guide.") + "\n\n",
            "</details>\n\n",
        ])
    if not exercises:
        body.append("No exercise headings were detected; use the teaching guide's self-checks.\n\n")
    body.append(
        "After solving, change one input or assumption and predict the result before\n"
        "running it. If a cell calls `project.solutions`, passing it verifies the\n"
        "reference demonstration; test your own implementation separately.\n"
    )
    return {"cell_type": "markdown", "id": "consolidated-hints", "metadata": {"tags": ["hints-section"]}, "source": body}


def process(nb_path: Path, refresh: bool = False) -> str:
    try:
        nb = json.loads(nb_path.read_text())
    except (ValueError, OSError) as error:
        return f"SKIP (parse error: {error})"
    indices = [i for i, cell in enumerate(nb["cells"]) if HINTS_MARKER in source(cell)]
    if indices and not refresh:
        return "skip (already present; use --refresh to regenerate)"
    exercises = find_exercises(nb)
    cell = make_hints_cell(exercises)
    if indices:
        old = nb["cells"][indices[0]]
        cell["metadata"] = old.get("metadata", cell["metadata"])
        cell["id"] = old.get("id", cell["id"])
        nb["cells"][indices[0]] = cell
        for i in reversed(indices[1:]):
            del nb["cells"][i]
    else:
        nb["cells"].append(cell)
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")
    return f"{'refreshed' if indices else 'appended'} ({len(exercises)} exercises)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="replace existing generated sections")
    args = parser.parse_args()
    notebooks = sorted(ROOT.glob("week-*/workbook.ipynb"))
    if not notebooks:
        print("no workbooks found", file=sys.stderr)
        return 1
    for notebook in notebooks:
        print(f"{notebook.parent.name}: {process(notebook, refresh=args.refresh)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
