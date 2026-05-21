#!/usr/bin/env python3
"""Generate a `week-NN-slug/` folder with boilerplate notes.tex, workbook.ipynb,
extended_reading.tex, project/, and tests/.

    python scripts/new_week.py 01 mdps "Markov Decision Processes"
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTES_TEMPLATE = r"""\documentclass[11pt,oneside]{scrbook}
\usepackage[a4paper,margin=2.4cm,marginparwidth=3.2cm,marginparsep=0.5cm]{geometry}
\usepackage{../common/rlstudy}

\title{{\color{rlblue}\textbf{Week %(n)s} \\ \Huge %(title)s}}
\author{STUDY-RL — a self-paced Deep RL curriculum}
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

\chapter{Introduction}
TODO.

\chapter{Mathematical formulation}
TODO.

\chapter{Algorithms}
TODO.

\chapter{Implementation walkthrough}
TODO.

\begin{safetyangle}
Safety angle: \emph{TODO}.
\end{safetyangle}

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

\title{Extended reading — Week %(n)s: %(title)s}
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
    "Each section ends with a graded **Exercise** with hints. Solutions live in the project module.\n"
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
TEST_TEMPLATE = """import importlib


def test_project_importable():
    importlib.import_module('week-%(n)s-%(slug)s.project')
"""


def main(n: str, slug: str, title: str) -> int:
    folder = ROOT / f"week-{n}-{slug}"
    if folder.exists():
        print(f"already exists: {folder}", file=sys.stderr)
        return 1
    (folder / "project").mkdir(parents=True)
    (folder / "tests").mkdir()
    ctx = {"n": n, "slug": slug, "title": title}
    (folder / "notes.tex").write_text(NOTES_TEMPLATE % ctx)
    (folder / "extended_reading.tex").write_text(EXTENDED_TEMPLATE % ctx)
    nb = json.dumps(WORKBOOK_TEMPLATE, indent=1)
    (folder / "workbook.ipynb").write_text(nb.replace("%(n)s", n).replace("%(title)s", title))
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
