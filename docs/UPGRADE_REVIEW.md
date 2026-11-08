# Textbook and statistics upgrade review

This November 2026 revision strengthens the mathematical explanations,
statistical experiments, notebooks, and reading experience across all 26 weeks.
The existing public address remains
[study-rl.samuelzhang.co.uk](https://study-rl.samuelzhang.co.uk/);
[study-rl.samuelzhang.dev](https://study-rl.samuelzhang.dev/) redirects to it,
retaining lesson routes and the existing browser-local progress origin.

The requested educational and visual targets are 9.8/10 and 9.5/10. The
[upgrade plan](UPGRADE_PLAN.md) defines the review criteria. These are
editorial design targets; this revision does not claim a measured learner
ability score or a human learning-outcome study.

## What changed

- Nine original [probability and statistics lessons](statistics/README.md)
  connect conditioning, expectation, uncertainty, comparisons, resampling,
  validation, importance weighting, adaptive data, and statistical claims.
  The [source catalog](statistics/SOURCES.md) identifies the supplied book
  editions and checked section-specific PDF pages.
- All 26 teaching guides now include distinct calculated statistical cases,
  interpretations, changed-assumption checks, and relevant foundation links.
  All 26 lectures and 26 reading companions connect those ideas to their topic.
  All 53 PDFs, including the syllabus, were rebuilt.
- All 27 workbooks, including the optional LLM lab, have a new investigation
  with a prediction, hand-worked anchor, executable plot, result-reading
  guidance, and hidden worked transfer feedback. Every original cell ID and
  position is preserved. The current source has 945 cells, including 441 code
  cells, with saved outputs and execution counts cleared.
- Eighteen scaffolded workbooks now explicitly select reference or learner
  implementations; Week 26 retains its existing no-fallback learner path.
  Twenty primitive checks gained exact numerical fixtures. Regression checks
  demonstrate that incorrect learner implementations fail without invoking
  the reference implementation.
- The website shows validated notebook figures, tables, and bounded text.
  Output provenance includes notebook and Python-dependency hashes. Missing,
  failed, or stale executions cannot pass the publication gate.
- Three new browser investigations teach sampling intervals, paired
  comparisons, and clustered resampling. Each provides its model assumptions,
  interactive calculations, accessible data, and CSV export.

The source notebooks stay editable and runnable locally. The browser displays
recorded Python results and runs its JavaScript instruments; it does not execute
the Python notebooks.

## Statistical repairs

The revisions align CVaR instructions with exact fractional tail mass, separate
soft KL penalties from hard constraints, and replace stochastic method rankings
as correctness checks with appropriate mathematical contracts and observed
comparisons. OPE rejects invalid propensities and reports undefined SNIPS
separately from a valid estimate of zero. Nuisance fitting in the Week 11
reference experiment uses independent synthetic fitting samples.

Related checks cover missing examination support in ranking, zero-weight ESS,
the distinction between probability masses and continuous densities, and
independent evaluation units. The Week 11 empirical figures repeat complete
logging experiments under a fixed environment and target; their data records
identify the sampling and fitting seeds, Monte Carlo reference uncertainty,
and finite-repetition error summaries.

Descriptive SD bands, standard errors, confidence intervals, empirical tail
averages, and safety guarantees are explained as different quantities. Patient,
prompt, session, episode, and training-run examples state which unit is repeated
and what remains fixed. Conditional fitted-model uncertainty is distinguished
from uncertainty of refitting and model selection.

## Independent review and evidence

Separate teaching, notebook/statistics, and website workstreams reviewed each
other's changes. The independent notebook review read all 27 new investigations,
checked numerical anchors and transfer feedback, inspected retained results,
and opened each investigation's plot individually at native resolution.
Corrections included clustering arithmetic, a shared-effect cancellation,
probability-axis labels, a selection-interval interpretation, and clipped plot
labels. No actionable mathematical, clarity, or presentation blocker remained
in that inspected appendix scope. This is a scoped review, rather than a new
formal verification of every theorem in the inherited course.

The textbook authoring checks covered 43 deterministic calculations and local
destinations in 37 Markdown documents. All 52 weekly PDFs passed text extraction
and final build-log checks without overflow or unresolved-reference warnings;
representative new pages were also rendered and inspected. Details appear in
the [textbook validation record](statistics/VALIDATION.md).

Reference execution runs literal notebook cells, in order, in isolated copied
course trees using fresh kernels. All 27 workbooks have completed all 441 code
cells without reduced training budgets or source overrides. The execution
manifest retains safe outputs, environment versions, and exact source/dependency
hashes. Regenerate it after any relevant source change.

Python checks cover shared helpers, every weekly suite in its own process,
the optional LLM lab, learning projects, and release tools. Browser model,
state, progress, diagnostics, safe-output, and freshness checks accompany the
rendered-content and download checks. All 26 extracted weekly bundles were
checked against canonical notebooks and their setup imports.

Fresh Chromium checks cover 12 original workflow groups, seven reader/progress
groups, and nine statistics/reading groups. Automated accessibility checks
performed 48 WCAG A/AA scans at 320, 390, and 1440 pixel widths with no reported
violations. Responsive checks additionally cover 768 pixels. Manual keyboard
and accessibility-tree inspection checked routing focus, reading order,
disclosures, material navigation, reader controls, and statistical sliders.
This is not a human screen-reader study or a fresh Firefox/WebKit audit.

Desktop and mobile screenshots were inspected. At 390 by 844 pixels, the
mathematical lesson begins within the first viewport; secondary resources and
reader settings remain available through native disclosures. The homepage's
illustration uses explicit invented run values and correctly distinguishes SD,
SE, and confidence intervals.

## Reproduce and publish

From the repository root, with the documented Python, Node, and TeX tools:

```bash
make course-tests llm-tests learning-tests lab-tests
make release-check
make portal-release
make portal-serve
```

Use the [portal browser instructions](../portal/README.md) for the dedicated
workflow, learning, and statistics suites. `make portal-release` executes the
notebooks, prepares the materials, builds, tests, and verifies the website in
sequence. `make portal-verify` checks current content, interface, execution,
and complete artifact fingerprints. Publication must use that exact verified
artifact. The deployed `release.json` and `deployment.json` identify the source
revision and artifact; a retained old build cannot validate newer materials.

Book PDFs and the instructor manual are local reference inputs, excluded from
Git, website assets, and student archives. Original prose and fixtures are
published with citations and [attribution](ATTRIBUTION.md).
