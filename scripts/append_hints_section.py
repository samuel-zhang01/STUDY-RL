#!/usr/bin/env python3
"""Append a 'Hints toward answers' section to every week-*/workbook.ipynb.

For each exercise we detect by markdown heading, emit a stronger nudge:
key equation to apply, expected output shape, sanity check, common pitfall,
and a pointer to project/solutions.py for the worked answer.

Idempotent — re-running won't duplicate the section. Run before papermill
in the Phase-4 sweep so notebooks re-execute with the augmented hints.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HINTS_MARKER = "<!-- AUTO-HINTS-SECTION -->"

# Generic per-exercise nudge — week-specific hints already live above each
# scaffold cell; the consolidated bottom section gives a denser cheat-sheet.
GENERIC_NUDGE = """
- **Key relation:** look up the equation in the corresponding `notes.pdf` chapter \"Mathematical formulation\". The notebook prompt names the function — find that function in `project/solutions.py` and skim its docstring.
- **Expected output shape:** scalar for losses / regret, `(n_steps,)` for training curves, `(nS,)` for value functions, `(nS, nA)` for Q-tables, dicts for hyperparameter sweeps.
- **Sanity checks (in order, cheapest first):**
  1. Run on the smallest fixture first (`n_steps=10`, `n_episodes=5`).
  2. Compare against the random-policy baseline you already have.
  3. Check finite values (no `nan` / `inf` after the first step).
  4. Plot the learning curve — if it's flat, recheck the step direction (gradient sign).
- **Common pitfalls for this exercise type:**
  - Off-by-one in advantage estimation (GAE / TD): always test on a deterministic toy first.
  - Forgetting `torch.no_grad()` around target-net targets — silent inflation of the loss.
  - Updating the target / behavior policy with the same gradient — double counting.
  - Using `.detach()` on the wrong tensor in DPO / PPO — kills the gradient.
  - In off-policy / OPE: importance weights must use the behavior policy at the time the action was logged (not the current policy).
- **Worked answer:** see the matching function in `project/solutions.py` — its body shows the canonical implementation; read it only after a serious attempt.
"""


def already_has_hints(nb: dict) -> bool:
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if HINTS_MARKER in src:
            return True
    return False


def find_exercises(nb: dict) -> list[str]:
    titles: list[str] = []
    pat = re.compile(
        r"^\s*#{1,4}\s*("
        r"(?:Exercise|Ex|E|Question|Q|Task|Problem)\s*\d+[^\n]*"
        r"|\d+\.\s+[A-Z][^\n]*"
        r")",
        re.IGNORECASE | re.MULTILINE,
    )
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        for m in pat.finditer(src):
            title = m.group(1).strip().rstrip("—-:").strip()
            if title not in titles:
                titles.append(title)
    return titles


def make_hints_cell(week_name: str, exercises: list[str]) -> dict:
    body = [
        f"{HINTS_MARKER}\n",
        "## Hints toward answers\n",
        "\n",
        "_A consolidated cheat-sheet — read only after you've attempted each exercise. "
        "Stronger than the inline hints above the scaffolds, weaker than the full solution. "
        f"Worked answers are in `project/solutions.py` in the same folder (`{week_name}`).\n_\n\n",
    ]
    if not exercises:
        body.append("> No exercise headings detected — open the notebook and add the relevant nudges manually.\n")
    else:
        for ex in exercises:
            body.append(f"### {ex}\n")
            body.append(GENERIC_NUDGE)
            body.append("\n")
    body.append("\n---\n_End of week. Mark this notebook in your tracker only after every exercise's hidden test cell passes._\n")
    return {
        "cell_type": "markdown",
        "metadata": {"tags": ["hints-section"]},
        "source": body,
    }


def process(nb_path: Path) -> str:
    try:
        nb = json.loads(nb_path.read_text())
    except Exception as e:
        return f"SKIP (parse error: {e})"
    if already_has_hints(nb):
        return "skip (already present)"
    exercises = find_exercises(nb)
    cell = make_hints_cell(nb_path.parent.name, exercises)
    nb["cells"].append(cell)
    nb_path.write_text(json.dumps(nb, indent=1))
    return f"appended ({len(exercises)} exercises detected)"


def main() -> int:
    weeks = sorted(ROOT.glob("week-*/workbook.ipynb"))
    if not weeks:
        print("no workbooks found", file=sys.stderr)
        return 1
    for nb in weeks:
        print(f"{nb.parent.name}: {process(nb)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
