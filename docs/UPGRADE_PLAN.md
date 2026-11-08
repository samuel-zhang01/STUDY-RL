# Textbook grounded course upgrade

The November 2026 upgrade covers all 26 weekly bundles, the optional LLM
workbook, probability and statistics prerequisites, and the teaching website.
The educational review target is 9.8/10; the visual review target is 9.5/10.
These are review thresholds, not measured learning outcomes.

The course should let a Python-capable learner rebuild the mathematics while
studying RL. Every added investigation should connect a concrete question to
its assumptions, a hand calculation, executable evidence, interpretation, and
a changed-assumption task. Existing useful derivations, diagnostic feedback,
progress backups, and exact section/cell links must keep working.

## Work and ownership

| Work | Required result |
|---|---|
| Textbook foundation | Original bridge lessons with verified book editions, sections, and section-specific pagination; no redistribution of source PDFs or instructor solutions. |
| Weekly explanations | Week-specific probability/statistics reasoning in all 26 guides, lectures, and reading companions; correct units, assumptions, examples, and empirical claims. |
| Notebooks and code | Correct estimator contracts and tail-risk definitions; meaningful learner checks; bounded statistical investigations in every week and the optional lab. |
| Notebook publication | Clean learner source notebooks plus separately executed reference artifacts; safe visible figures/tables with source and execution provenance. |
| Website | A coherent scientific workbook with readable hierarchy, useful lesson progression, contextual foundations, interactive statistics, accessible routing, and mobile reading. |
| Deployment | One validated static artifact, reversible release, valid HTTPS, and live checks at both requested domains. |

## Educational review

| Criterion | Weight |
|---|---:|
| Correctness, assumptions, and units | 25 |
| Prerequisite progression | 20 |
| Explanatory clarity | 20 |
| Worked calculations and code alignment | 15 |
| Statistical interpretation | 15 |
| Retrieval and transfer | 5 |

Target 98/100. Any unresolved mathematical error, misleading empirical claim,
broken essential learning link, or reference-only learner check prevents
acceptance regardless of the total. Reviews must state which materials were
actually read and tested. Learner comprehension requires learner evidence;
reviewer ratings alone cannot demonstrate it.

## Visual review

| Criterion | Weight |
|---|---:|
| Typography and hierarchy | 25 |
| Reading and navigation | 20 |
| Diagrams, figures, and outputs | 20 |
| Responsive use and accessibility | 20 |
| Consistency and performance | 15 |

Target 95/100. Inspect home, curriculum, mathematical lessons, statistics,
notebook results, revision, progress, and downloads at 320, 390, 768, and 1440
pixel widths. Automated checks supplement manual keyboard, focus, zoom, and
representative screen-reader checks. No serious or critical accessibility
finding may remain open.

## Release evidence

Execute reference notebooks in isolated week-specific kernels and retain
portable outputs. State any bounded training overrides beside the results;
do not describe these as full benchmark reproductions. Check plotted values,
signs, interval definitions, and conclusions against actual execution.

Rebuild changed PDFs and the complete website snapshot before running release,
content, numerical-model, download, and browser checks. Validate the exact
artifact that is published and retain source/output hashes. Keep dated
validation evidence separate from historical reviews.

Update the existing deployment in place. The canonical address remains
`https://study-rl.samuelzhang.co.uk/`. Add `https://study-rl.samuelzhang.dev/`
as a redirect while retaining lesson routes. Do not create the previously
proposed `rl.*` hostnames. Keeping the existing canonical origin preserves
browser-local learning progress. Relevant DNS, TLS, and serving configuration
need fresh checks through the authenticate and stack-verify workflows before
release.

## Scoping baseline

The initial audit inventoried 26 weekly notebooks and validated their schemas;
all 829 cells were output-free, consistent with the existing source policy.
Substantive review sampled foundations, PPO, OPE, safety, and causal material.
No complete notebook execution or fresh visual review was performed in that
scoping pass. The initial portal unit suites passed. Identified repairs include
CVaR definition drift, invalid OPE failure states, unsupported empirical claims,
and the notebook renderer's discarded outputs. Final acceptance requires new
evidence after these changes.
