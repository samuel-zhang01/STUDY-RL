# Public-release preparation review

Prepared on **6 October 2026**. This review covers the release tree and locally
sanitized Git history. The sanitized history has been pushed to GitHub and
verified in a fresh mirror clone. The repository remains private because
GitHub still serves original content through cached commit-SHA views.

The **9 October interactive teaching revision** adds a staged visual lab,
standard-library games, genuine learner exercises, and bridges across all
26 weeks. Its [validation record](INTERACTIVE_VALIDATION.md) reports 462 code
checks, browser workflows and independent student scores. The original PDF
and history audit below remains the 6 October record; unchanged PDFs were not
rebuilt for the later Markdown/browser revision.

## Audit scope and findings

The initial audit inspected all 30 commits reachable from the fetched branch,
1,078 unique file blobs, commit metadata, notebook sources/outputs, extracted
historical PDF text, and metadata from 298 historical PDF versions.

[Gitleaks 8.30.1](https://github.com/gitleaks/gitleaks) found **no credentials**
in either the complete history patch scan or the extracted historical text
and PDF scan. No private network addresses or machine-specific home paths
were found in those historical text scans. These are detection results within
that scope, not a guarantee that every possible secret format is recognized.

The privacy review did find committed local assistant permissions, personal
email addresses in commit metadata, saved notebook executions, and named
private-system architecture descriptions. The release preparation:

- Removes local assistant settings and generated notebook executions from Git.
- Replaces private application names with neutral synthetic-example names.
- Removes internal source paths, service details, private scoring descriptions,
  and unsupported production-deployment claims from the ATS narrative.
- Rebuilds the corresponding PDFs and application charts, rather than leaving
  their earlier text in shipped binary artifacts.
- Removes 259 flagged historical blob versions and rewrites personal commit
  email addresses to non-personal noreply addresses. Contributor names and
  attribution are retained; some empty historical commits may be pruned.
- Renames the ATS week throughout history so its directory name does not
  disclose the former private project name.

Original files and history are preserved in a private backup outside this
checkout. That backup contains the original sensitive material and is excluded
from the public release. Historical revisions affected by content removal are
not guaranteed to be complete runnable snapshots.

The history cleanup uses
[git-filter-repo](https://github.com/newren/git-filter-repo). The remote branch
has been replaced using an explicit force-push lease. Cached original content
still needs to be purged before making the repository public;
[GitHub's sensitive-data removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
explains the additional remote and cached-reference considerations.

## Repository changes

Learner documents now live in `docs/`; earlier teaching reviews and dated TSV
records live in `docs/archive/`. The canonical syllabus is
`syllabus/syllabus.pdf`. Week 25 is `week-25-synthetic-ats-rl-audit`.
Weekly directory placement is retained because the projects and notebooks use
local imports relative to the repository root.

The README includes a first-week walkthrough, isolated course and LLM
installation paths, kernel registration, selected-week and full-course tests,
PDF/notebook build instructions, troubleshooting, and scope boundaries.
Original contributions use MIT; third-party attribution and model terms are
covered in `THIRD_PARTY_NOTICES.md`.

The default requirements now target the bundled CPU projects, including the
Box2D environments used by the DQN/Rainbow exercises. SWIG is installed before
the requirements when compiling Box2D bindings. Optional paper
benchmark dependencies are in `requirements-extras.txt`, and the LLM lab keeps
its separate tested dependencies. Credentials, local settings, private keys,
model weights, and generated reports are ignored. `make clean` removes only
ignored caches and preserves tracked files, shipped PDFs, and environments.

CI definitions run isolated CPU tests, release-file checks, and a redacted
full-history credential scan. GitHub Actions are pinned to commit IDs and the
Gitleaks executable is pinned to a release with a fixed SHA-256 checksum.
The first fresh Linux course run caught a missing Box2D extra that the existing
macOS environment had masked. The course requirements and SWIG setup were
corrected. At code revision `30fc1f6`, both
[course CI](https://github.com/samuel-zhang01/STUDY-RL/actions/runs/37489594099)
and the [history secret scan](https://github.com/samuel-zhang01/STUDY-RL/actions/runs/37489594637)
passed; the fresh Linux course run passed all 428 tests.

## Validation

| Check | Result |
|---|---|
| Isolated test suites | **428 passed**: all 26 weeks, shared utilities, maintenance tools, and the offline LLM lab |
| Changed weekly workbooks | Weeks 1, 11, and 25 execute in fresh kernels with the correct working directory |
| Optional LLM numerical workbook | Executes in a fresh kernel without pretrained weights |
| Source notebooks | All 27 validate; code parses; source notebooks contain no saved execution outputs |
| Python and documentation | Release-file checker passes; local Markdown and notebook links resolve |
| Lecture/reading builds | All 52 weekly PDFs and the syllabus compile |
| PDF layout | All 53 main PDFs pass the raster edge and compile-log checks; affected pages and charts were visually reviewed |
| Shipped PDF privacy | All 138 shipped PDFs checked for former private names, internal paths, personal addresses, and home paths |
| Core installation | Existing macOS environment and fresh Linux CI installation pass; SWIG/Box2D included, JupyterLab CLI and dependency consistency checked |
| Repository hygiene | `git diff --check` passes; cleanup safeguards tested |

Validation used macOS/Apple Silicon, Python 3.12.15, NumPy 2.5.2, SciPy 1.18.1,
PyTorch 2.14.0, Gymnasium 1.3.0, pytest 9.1.1, Transformers 5.16.1, and
PEFT 0.20.0. CPU thread counts were limited for tests and plots. Existing
non-failing PyTorch/Gymnasium/PEFT warnings remain visible in the test output.

The core install was checked in the existing macOS environment and a fresh
Linux CI installation. The Linux run used Python 3.12.14, PyTorch 2.14.1+cpu,
Gymnasium 1.4.0, Box2D 2.3.10, NumPy 2.5.3, and SciPy 1.18.1. GPU training, optional
large benchmark environments, pretrained-model quality, every bibliography
entry, and production/clinical safety were not revalidated for this release.
Earlier measured LLM results remain documented in
[the lab validation report](../llm-lab/VALIDATION.md).

## Reproduce the checks

From the repository root with the relevant environments active:

```bash
make course-tests PY=python
make llm-tests PY=python
make release-check PY=python
make secrets
make syllabus weeks
python scripts/pdf_margin_sweep.py --render
make llm-notebook PY=python
```

The [README](../README.md) documents installation and selected-week notebook
execution. Newly generated validation tables stay under ignored `reports/`;
[archived records](archive/README.md) describe earlier revisions.

## Final local history verification

After rewriting, the reachable history contains **27 commits**, **953 file
versions**, and **245 PDF versions**. The release tree was byte-for-byte
unchanged by the filter. A second complete content/metadata pass found no
remaining flagged private names, private source references, personal email
addresses, or home paths. Both the Git patch scan and the extracted historical
text/PDF credential scan returned no Gitleaks findings.

All **259 flagged original blob objects** are absent from the cleaned local
object database. Local application diff-snapshot refs were also backed up and
removed so they cannot retain the old tree. Only the sanitized `main` ref is
intended for publication. A fresh GitHub mirror contains none of the flagged
original objects and passes both secret scans and the complete source/PDF/
metadata privacy checks. GitHub cached original content and the private local
backup still retain the old material. A private GitHub Support cache-purge
request has been prepared but has not been submitted.
