#!/usr/bin/env python3
"""Generate a `week-NN-slug/` folder with boilerplate teaching_guide.md, notes.tex, workbook.ipynb,
extended_reading.tex, project/, and tests/.

    python scripts/new_week.py 01 mdps "Markov Decision Processes"
"""
from __future__ import annotations
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTES_TEMPLATE = r"""\documentclass[11pt,oneside]{scrbook}
\usepackage[a4paper,margin=2.4cm,marginparwidth=3.2cm,marginparsep=0.5cm]{geometry}
\usepackage{../common/rlstudy}

\title{{\color{rlblue}\textbf{Week %(n)s} \\ \Huge %(title)s}}
\author{STUDY-RL --- a self-paced Deep RL curriculum}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

\chapter*{Learning objectives}
\addcontentsline{toc}{chapter}{Learning objectives}

\begin{keyidea}
By the end of this lecture you will be able to: \emph{(to be authored)}.
\end{keyidea}

%% ====================== SECTIONS BELOW (authored per week) ======================

\chapter{Motivation \& intuition}
TODO.

\chapter{Mathematical formulation}
TODO.

\chapter{Algorithms}
TODO.

\chapter{Implementation walkthrough}
TODO.

\chapter{Empirical results}
TODO: generate a deterministic project figure and include it here.

\chapter{Safety angle}
\begin{safetyangle}
Safety angle: \emph{TODO}.
\end{safetyangle}

\chapter{Matching application}
\begin{matching}
Matching application: \emph{TODO}.
\end{matching}

\chapter{Exercises}
See \texttt{workbook.ipynb} for graded exercises with hints.

\chapter{Further reading}
See \texttt{extended\_reading.pdf}.

\end{document}
"""

EXTENDED_TEMPLATE = r"""\documentclass[11pt]{scrartcl}
\usepackage[a4paper,margin=2.6cm]{geometry}
\usepackage{../common/rlstudy}

\title{Extended reading --- Week %(n)s: %(title)s}
\date{\today}
\begin{document}
\maketitle

\section*{Papers}
\begin{itemize}
  \item TODO
\end{itemize}

\end{document}
"""

WORKBOOK_TEMPLATE = {
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": [
    "# Week %(n)s — %(title)s · Workbook\n",
    "\n",
    "Start with the [teaching guide](teaching_guide.md) for intuition, derivations, and worked examples.\n",
    "Each section ends with an **Exercise** with hints. Attempt it before opening the reference solutions in the project module.\n"
  ]},
  {"cell_type": "code", "metadata": {}, "source": [
    "import sys, os\n",
    "sys.path.insert(0, os.path.abspath('..'))\n",
    "from common.plotting import use_rl_style, RL_PALETTE\n",
    "from common.seed import seed_everything\n",
    "use_rl_style(); seed_everything(0)\n"
  ], "outputs": [], "execution_count": None},
  {"cell_type": "markdown", "metadata": {}, "source": [
    "## Exercise 1 — TODO\n",
    "*Hint: TODO*\n"
  ]}
 ],
 "metadata": {
   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
   "language_info": {"name": "python", "version": "3.11"}
 },
 "nbformat": 4, "nbformat_minor": 5
}

PROJECT_INIT = '"""Mini-project for week %(n)s — %(title)s."""\n'
GUIDE_TEMPLATE = """# Week %(n)s — %(title)s: teaching guide

Read [the course study guide](../docs/STUDY_GUIDE.md) and [notation](../docs/GLOSSARY.md).
This is an authoring scaffold; complete and verify it before teaching from it.

## Prerequisites and the question

TODO: name earlier concepts and the new problem they cannot yet solve.

## Analogy and its limits

TODO: map each object to a mathematical concept and state where it stops fitting.

## Derivation

TODO: define symbols, timing, data distribution, objective, and assumptions;
justify each important step; label approximations and proof sketches.

## Worked example 1

TODO: setup, prediction, intermediate calculations, final result, interpretation.

## Worked example 2: change an assumption

TODO: use an edge case or counterexample; calculate and verify the result.

## From equations to code

TODO: link real functions in project/ and state the implementation's limits.

## Self-checks

TODO: ask one retrieval, one calculation, and one transfer question.

<details>
<summary>Reveal answers after attempting the questions</summary>

TODO: explain the answers, including why tempting alternatives fail.

</details>

## Sources and arithmetic validation

TODO: cite primary sources and record the numerical checks actually run.
"""

TEST_TEMPLATE = """import importlib


def test_project_importable():
    importlib.import_module('week-%(n)s-%(slug)s.project')
"""


def main(n: str, slug: str, title: str) -> int:
    if not n.isdigit() or not 1 <= int(n) <= 99:
        print("week number must be an integer from 1 to 99", file=sys.stderr)
        return 2
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        print("slug must contain lowercase letters/digits separated by hyphens", file=sys.stderr)
        return 2
    n = f"{int(n):02d}"
    folder = ROOT / f"week-{n}-{slug}"
    if folder.exists():
        print(f"already exists: {folder}", file=sys.stderr)
        return 1
    (folder / "project").mkdir(parents=True)
    (folder / "tests").mkdir()
    ctx = {"n": n, "slug": slug, "title": title}
    tex_escape = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                  "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                  "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    tex_ctx = {**ctx, "title": "".join(tex_escape.get(c, c) for c in title)}
    (folder / "notes.tex").write_text(NOTES_TEMPLATE % tex_ctx)
    (folder / "teaching_guide.md").write_text(GUIDE_TEMPLATE % ctx)
    (folder / "extended_reading.tex").write_text(EXTENDED_TEMPLATE % tex_ctx)
    nb = copy.deepcopy(WORKBOOK_TEMPLATE)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"week-{n}-cell-{i}"
        cell["source"] = [line.replace("%(n)s", n).replace("%(title)s", title)
                          for line in cell["source"]]
    (folder / "workbook.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (folder / "project" / "__init__.py").write_text(PROJECT_INIT % ctx)
    (folder / "tests" / "__init__.py").write_text("")
    (folder / "tests" / f"test_smoke.py").write_text(TEST_TEMPLATE % ctx)
    print(f"created {folder}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: new_week.py NN slug 'Title'", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
