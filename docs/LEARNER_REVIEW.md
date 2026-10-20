# Adversarial learner review — 20 October 2026

Acceptance target: **9.5/10**, with no unresolved mathematical, missing-material,
misleading-feedback or essential-navigation defect in the audited student path.
This is a multi-agent inspection and executable workflow audit, not a study of
human learning outcomes. The validation record below identifies the checked paths and remaining limits.

The reviewed path is: first visit → prerequisite/route → teaching companion →
full HTML lesson → visual investigation → workbook preview → complete download →
local implementation → evidence of understanding. The score covers this path
and the inspected explanations; it does not certify every claim in 52 lecture
and further-reading documents or imply that every research experiment was run.

## Findings addressed in this revision

The website now names its GitHub repository prominently, explains the first
study session, and links every week to its actual notebook, project code and
complete ZIP. The browser Python runtime and its duplicate practice interface
have been removed. All 26 guides include a topic-specific notebook investigation:
prediction, exact implementation, changed input, observable result and a limit
on the conclusion. Notebook introductions distinguish supplied reference code
from a student's own work.

Four richer ECharts instruments expose seeded exploration, synchronous Bellman
updates, signed PPO geometry and importance weights. They include numerical
readouts, tables, CSV export, accessible controls and reduced-motion behavior.
Ten courier investigations and eight existing mechanism demonstrations provide
shorter stepping stones. Reading navigation, search, mobile menus and progress
backups connect these parts.

The teaching review repairs unsupported guarantees and mislabelled metrics in
sampled notebooks and the week 8/week 21 lecture/readings. It distinguishes
empirical line search from TRPO's ideal guarantee, reference execution from
independent implementation, finite-sample behavior from population support,
feature-flip sensitivity from causal fairness, and percentile intervals from
intervals required to contain their point estimate.

## Acceptance result: 9.5/10 for the inspected learning journey

The teacher/content reviewer and the independent visual/usability reviewer
both meet the **9.5/10** threshold. A third reviewer independently completed
the first-visit, workbook, plot-control and download journey on desktop and
320px mobile, including a numerical check of both negative-advantage PPO
branches. These are expert rubric judgments, not estimates of student success.

| Review angle | Result and basis |
|---|---|
| Teaching and notebook connection | **9.5/10**: all 26 guides and notebook bridges inspected; topic-specific prediction, changed-input investigation, explanation and limits; targeted inaccurate claims repaired |
| Visual mathematical fidelity | **9.7/10**: seeded opportunity-cost accounting, synchronous backups, terminal/discount endpoints, signed PPO branches and slopes, IPS/SNIPS/ESS and support cases checked against independent numbers |
| Visual explanation and handoff | **9.5/10**: four instruments connect prediction, control, worked calculation, table/CSV, boundary case, primary source and actual notebook |
| Navigation and visual clarity | **9.5/10**: newcomer route, full HTML lessons, responsive selectors, named GitHub repository, direct notebooks and weekly bundles; clipped labels and mobile overflow repaired |
| Accessibility and resilience | **9.5/10**: native keyboard controls, focusable scroll regions, reduced motion, finite playback, route cleanup and chart-library fallback; automated and manual checks below |

## Executed validation for this revision

| Check | Observed result |
|---|---|
| Rendered course | **435 readable documents**, 26 complete HTML lessons, 26 companions, 26 weekly notebooks; zero equation errors or unresolved source links |
| Rendered-content, state and new-plot tests | **15 passed**, including every local document anchor and figure path |
| Independent integrated browser journeys | **12 groups passed**: all 26 guides/lectures, notebook/ZIP links, source/PDF reader, search, repeated section links, 22 experiments, progress backups and mobile layouts |
| Accessibility | **25 route/viewport/scheme audits**, zero axe WCAG A/AA violations; desktop, 320/390/768px, courier iframe interiors and dark-preference embedding checked |
| Browser diagnostics | Zero JavaScript errors and zero failed requests during the integrated suite |
| Weekly/shared Python suites | **346 passed across 28 isolated suites** |
| Offline LLM tests | **83 passed**; no pretrained-model download or GPU training |
| Standard-library coding projects | **21 passed** |
| Combined numerical and state models | **31 passed**, including existing courier cases, smaller mechanism labs and the new plots; these overlap the portal model/state tests above and must not be added as unique cases |
| Targeted teaching calculations | **15 fixture groups**, seven selected notebook-check cells, four constitutional boundary cases and week 21 regression cases pass |
| Rebuilt teaching PDFs | Four PDFs rebuilt; no undefined references or overfull boxes in those builds |
| Extracted downloads | **26/26 bundles** match canonical notebook/PDF/helper bytes; all 26 notebook setup cells import successfully in separate extracted directories |
| Source hygiene | Release checker and whitespace checks pass; redacted Gitleaks source scan found no credentials; dependency audit reports zero known vulnerabilities |

Download validation also extracts the actual hosted bundles, compares canonical
notebook/PDF/helper bytes, and imports each notebook's setup in an isolated
process. It caught Week 15's cross-week logger dependency, which is now included.
The Week 18/19 bundles also include the optional recipe's separate LLM-lab code
and small teaching fixtures. Python dependencies still need local installation.

## Remaining limits

The audit does not certify every historical claim in all 52 lecture/readings,
run every training notebook or large benchmark, or establish production safety.
Reading reference cells and passing supplied reference checks do not demonstrate
independent implementation. Automated accessibility checks complement the
manual keyboard review; they do not establish compatibility with every assistive
technology. A study with human learners remains the next source of evidence
about pacing and learning outcomes. There are no known unresolved veto defects
in the inspected path after the repairs above.

The [9 October review](archive/LEARNER_REVIEW_2026-10-09.md) and
[earlier interactive validation](archive/INTERACTIVE_VALIDATION_2026-10-09.md)
remain archived with their original scope and dates.
