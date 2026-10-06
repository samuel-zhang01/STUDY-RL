# Per-week bundle specification (read this before authoring any week)

Each week lives in `week-NN-slug/` and ships the lecture, workbook, reading,
project/tests, and a **teaching guide**. These are authoring targets, not a claim
that every existing bundle already satisfies every requirement below.

## 0. `teaching_guide.md` — the learner's entry point

- State the prerequisite ideas and the question that motivates this week.
- Use an analogy with an explicit mapping to mathematical objects. Explain
  where the analogy breaks; do not use it as a proof or a safety guarantee.
- Derive one central result step by step. Define symbols, indexing, objective,
  data distribution, and assumptions before using them. Name the justification
  for each important equality or inequality. Mark approximations and proof
  sketches explicitly; a cited theorem is not a proof supplied by the course.
- Include at least **two fully calculated numerical examples**: setup,
  prediction prompt, substitution, intermediate values, answer, interpretation.
  Include an edge case or counterexample, and verify the arithmetic in Python.
  Label invented numbers as teaching fixtures, not empirical measurements.
- Link formulas to actual project files/functions. State which paper components
  the mini-project omits, and which claimed guarantees do not apply to it.
- Finish with at least three self-checks: retrieve, calculate, and transfer to
  a changed assumption. Put answers behind `<details>` blocks.
- Link the guide near the start of `workbook.ipynb` and from `STUDY_GUIDE.md`.
- Treat page counts and callout counts as layout guidance. Do not pad a week or
  promote a sketch to a proof to satisfy a quota.

## 1. `notes.tex` → `notes.pdf` (Tufte-flavored lecture)

- Document class: `scrbook` (works with the included style file). Single-sided, A4, wide
  margins (already configured by `\usepackage{../common/rlstudy}`).
- **Length target: 25–45 pages compiled.** Dense but readable.
- Required structure (use the exact section names so the syllabus stays coherent):
  1. `\chapter*{Learning objectives}` — 4–7 explicit, measurable objectives in a
     `keyidea` box.
  2. `\chapter{Motivation \& intuition}` — analogies, anchoring to ML/DL the reader
     already knows. Use `marginnote` for sidenotes.
  3. `\chapter{Mathematical formulation}` — definitions in `definition` envs, key
     theorems in `theorem` envs **with full step-by-step derivations**; proofs in
     `proof` envs (sketch is OK when explicitly labelled). At least one TikZ figure
     and one `pgfplots` plot.
  4. `\chapter{Algorithms}` — pseudocode + worked numerical example. Use
     `algorithm2e` or a clearly formatted `tcolorbox`.
  5. `\chapter{Implementation walkthrough}` — minted code blocks showing the core
     PyTorch / NumPy implementation, cross-referenced to the `project/` folder.
  6. `\chapter{Empirical results}` — at least one figure of training-curve from the
     mini-project, with `\input{fig_*.pgf}` or `\includegraphics{fig_*.pdf}` (you
     generate these via `project/make_figures.py`).
  7. `\chapter{Safety angle}` — wrap the discussion in a `safetyangle` env. Concrete
     failure modes, mitigations, and 1–2 citations.
  8. `\chapter{Matching application}` — wrap in a `matching` env. Tie the lecture to
     synthetic insurance lead–market matching **or** recruitment candidate matching
     (synthetic ATS). Even toy bandit weeks must include one paragraph of applied
     translation.
  9. `\chapter{Exercises}` — list of 5–10 graded exercises, each tagged ★/★★/★★★ by
     difficulty, with hints in `intuition` envs. Exercises are *fully spelled out* in
     `workbook.ipynb`; here we summarise.
  10. `\chapter{Further reading}` — 3–5 sentences pointing at `extended_reading.pdf`.

- **Style mandates**: every key result derived step-by-step (no "after some algebra…");
  every algorithm given pseudocode + complexity; at least one named *Common pitfall*
  per chapter (`pitfall` env).
- **Compile rule**: `latexmk -pdf -shell-escape -interaction=nonstopmode notes.tex`.
  All Unicode must use TeX macros (`$\to$`, `$\leftrightarrow$`, never literal arrows
  in normal text). Math goes in `$…$` or `\[…\]`.

## 2. `workbook.ipynb` (Jupyter, graded, with hints)

- Kernel: `python3`. Insert at the top:
  ```python
  import sys, os
  sys.path.insert(0, os.path.abspath('..'))
  from common.plotting import use_rl_style, RL_PALETTE
  from common.seed import seed_everything
  use_rl_style(); seed_everything(0)
  ```
- **At least 8 exercises**, each ordered as prompt (markdown), optional
  collapsible hint, scaffold (code), and checks (code). Use `<details>` for
  collapsible hints; a blockquote alone is always visible. Checks must call
  the learner's function on meaningful inputs. Comparing the reference solution
  with itself does not validate a learner implementation.
- Exercises must escalate: warm-up (★), core (★★), and an *open-ended* one (★★★)
  per week that lets the learner explore.
- One **"meaningful objective"** exercise per week that connects to matching/ranking
  or safety (e.g. *Estimate the off-policy value of a synthetic insurance ε-greedy lead
  router using SNIPS*).
- The distributed notebook must have a clearly explained execution path using
  **reference solutions** in `project/solutions.py`. Label any fallback to a
  reference implementation. A separate learner attempt can use
  `# YOUR CODE HERE` and raise `NotImplementedError`; say that this incomplete
  attempt is expected to stop. Do not call a reference run a graded submission.
- Use a fresh kernel and the week directory as the execution working directory.
  Preserve existing notebook cell IDs, metadata, and outputs on targeted edits.
- Plots use `RL_PALETTE`; figures saved to `figs/` so the LaTeX can include them.

## 3. `extended_reading.tex` → `extended_reading.pdf`

- Document class: `scrartcl`. 6–12 pages.
- For each of **3–5 papers**: full citation (title, authors, year, venue, arxiv ID),
  one-paragraph summary, a "*Why it matters for this week*" line, and one
  *annotated key passage* (block quote + your 2–3 sentence commentary).
- Verify passages against the primary source and record the section/page. Keep
  direct quotations within 25 words per source, or use a clearly labelled
  paraphrase with commentary. Never present an invented summary as a quotation.
- End with a *Sourced from*: link to the research dossier in `research/`.

## 4. `project/` and `tests/`

- `project/` is a real Python package: `__init__.py`, `solutions.py` (reference
  implementations of every workbook exercise), `agent.py` / `env.py` / `train.py`
  as appropriate.
- `project/train.py` must implement `train(seed=0, **kwargs)` and a `__main__`
  guard. It should converge to a *target metric* in **under 60 seconds on CPU**
  (or under 5 minutes on a single consumer GPU for the deep-RL weeks).
  Document the target in the module docstring.
- `project/make_figures.py` generates the figures used in `notes.pdf` (saved to
  `figs/*.pdf`). It must be deterministic given a seed.
- `tests/test_smoke.py` runs a 1–5 second smoke training pass and asserts a sane
  metric (e.g. return ≥ random baseline).
- `tests/test_solutions.py` calls each solution function on a tiny fixture and
  checks the contract that `workbook.ipynb` expects.

## Quality bar (every week must pass these checks before commit)

```bash
# 1. PDF compiles
latexmk -pdf -shell-escape -interaction=nonstopmode notes.tex && \
latexmk -pdf -shell-escape -interaction=nonstopmode extended_reading.tex

# 2. Notebook executes end-to-end (uses the reference solutions)
papermill --kernel python3 workbook.ipynb workbook.executed.ipynb

# 3. Tests pass
PYTHONPATH=. pytest week-NN-slug/tests -q
```

## Repository conventions

- Folder: `week-NN-slug` (zero-padded N, kebab-case slug).
- Importable as `from week_NN_slug.project import ...` is **not required**; tests use
  `pytest` from the repo root with `PYTHONPATH=.`. Inside week tests use
  `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` or rely on
  pytest's rootdir discovery.
- All RL primitives that aren't week-specific live in `common/` and are imported.
- Random seeds: use `common.seed.seed_everything`. Default seed is `0`.
- Save heavy artifacts in `runs/` (already gitignored).

## Safety + matching callouts — required

- **Every** week's `notes.tex` has a `safetyangle` chapter (even foundations weeks
  — there's always *some* safety angle, e.g. "exploration in a high-stakes domain
  means you can't afford epsilon-greedy without bounds").
- **Every** week's `notes.tex` has a `matching` chapter linking back to either
  synthetic insurance lead↔market matching or ATS (candidate↔job) or both. Some weeks lean
  hard on this (11, 12, 24) — others can be a single page.

## Citations + dossier integration

- Each week MUST cite the SOTA dossier sections relevant to it. The bibliography
  for `notes.tex` and `extended_reading.tex` can be inline (`\textsc{Schulman et
  al.\ 2017, arXiv:1707.06347}`) — no global `.bib` required.

## Done means

- All included weeks (currently 26) pass the three quality checks above.
- `make qa` exits 0.
- Every week shows a non-empty learning curve figure in `notes.pdf`.
- The teaching guide's arithmetic and local links have been checked, and a
  learner can identify assumptions, reconstruct an update, and try a changed
  input before revealing the answer.
- Report what was actually run, with failures and missing dependencies. File
  existence alone is not evidence of execution, convergence, or understanding.
