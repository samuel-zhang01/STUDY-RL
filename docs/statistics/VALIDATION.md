# Statistics teaching validation — 8 November 2026

This record concerns the new probability/statistics bridge and its weekly integrations. It does not replace the older course reviews or claim measured human learning outcomes.

## Authored scope

- Nine original foundation lessons, an entry route and a textbook source catalog.
- A distinct statistical fixture and changed-assumption task in every one of the 26 weekly teaching guides, mirrored in each lecture and connected to each reading pack.
- Explicit distinctions among outcome SD, mean SE, interval coverage, evaluation noise, training variation, pairing, clustering, missing support and conditional versus full-pipeline uncertainty.
- Corrections to Week 1 empirical captions, Week 3/11 helper excerpts and Week 20's overgeneralized no-SFT explanation. The syllabus points to the actual study-guide location and the statistics repair route.

## Executed checks

| Check | Observed scope and result |
|---|---|
| Numerical calculations | 43 deterministic Python/NumPy/SciPy checks passed, covering all 26 new weekly core fixtures and central calculations across all nine foundation lessons |
| Interval arithmetic | SciPy independently reproduced the four-unit paired intervals: approximately `[0.58131, 2.41869]` in Lesson 4 and `[-2.43743, 4.43743]` in Week 9 |
| Markdown destinations | All local link destinations in the 11 initial foundation/catalog files and 26 weekly teaching guides existed |
| Existing navigation | Every pre-existing weekly guide H2 heading was retained, preserving its existing section anchor |
| PDF builds | All 26 lectures, 26 reading packs and the syllabus rebuilt successfully with `latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error`; affected lectures rebuilt again after helper-alignment corrections |
| PDF content | Text extraction from all 52 weekly PDFs found their new statistics sections |
| PDF layout | Final TeX logs had no overfull-box or undefined-reference/citation warnings; representative new pages in Weeks 1, 20 and 26 were rendered and visually inspected |
| Textbook provenance | Section headings, source addresses and one-based PDF pages were checked in the supplied editions; the Ross textbook/instructor-manual concatenation is explicitly recorded |

The numerical checks validate the stated fixtures under their assumptions. They do not validate those assumptions in a deployment. Representative PDF inspection is not a page-by-page visual audit of every historical figure.

## Remaining evidence boundaries

The current editorial work is not a new reproduction of every historical training curve, every modern research theorem, or a human learner study. The new fixtures are labeled as invented examples. The supplied textbooks support probability and statistical reasoning; modern RL, LLM training and causal identification retain their own primary references and assumptions. Website, notebook execution and hosting have separate validation records and should be assessed against their actual executed checks.
