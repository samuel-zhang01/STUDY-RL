# Contributing

Corrections to explanations, reproducible bug reports, and focused improvements
to the exercises are welcome. Open an issue describing the week, file, expected
result, observed result, and a small reproduction before proposing a large change.

## Local setup

Use Python 3.12 and run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make course-tests PY=python
make release-check PY=python
```

For changes to the optional lab, install `llm-lab/requirements.txt` in a separate
environment and run `make llm-tests PY=python`. Lab tests use random local models
and synthetic fixtures; they do not require pretrained-model downloads.

## Make a focused change

1. Create a branch and keep the change scoped to one problem.
2. Follow the [week authoring specification](docs/WEEK_BUNDLE_SPEC.md).
   Define symbols and assumptions, cite primary sources, and distinguish toy
   simulations from paper reproductions and deployment evidence.
3. Run the affected weekly test suite in its own process:
   `python -m pytest week-01-mdps-bandits/tests -q`.
   Weekly bundles reuse local package names; collecting multiple weeks together
   is deliberately rejected. Use `make course-tests` to check all weeks.
4. For a notebook change, execute it with the week directory as the working
   directory, then commit the source notebook with outputs and execution counts
   cleared. Generated `workbook.executed.ipynb` files stay local.
5. For a TeX change, rebuild the affected PDF with the command in the README.
   Inspect its pages and run `python scripts/pdf_margin_sweep.py --render` after
   rebuilding all PDFs if you need a complete layout report.
6. Run `make release-check PY=python`, `make secrets`, and `git diff --check`.
   Explain the resulting behavior and the checks actually run in your pull request.

Keep credentials, private system descriptions, real applicant/patient records,
model weights, and machine-specific settings out of commits. Generate fixtures
locally with deterministic seeds. Cite third-party sources and retain their
licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).

## New weekly bundles

```bash
python scripts/new_week.py 27 topic-slug "Lesson title"
```

This creates a scaffold. Complete and verify it before adding it to the
curriculum index; a generated skeleton is not a finished lesson.
