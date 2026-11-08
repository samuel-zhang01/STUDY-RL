# STUDY-RL teaching portal

[Open the course](https://study-rl.samuelzhang.co.uk/) ·
[Get notebooks and exercises](https://study-rl.samuelzhang.co.uk/#/downloads) ·
[Source repository](https://github.com/samuel-zhang01/STUDY-RL)

The portal is the course's main entrance. **Start here** explains the first
session. Every week has a teaching companion, complete HTML lecture, readable
workbook, further reading and project code. All 26 weekly ZIP downloads include
the notebook's project and shared helpers. Python runs in the student's local
Jupyter environment; the website hosts explanations and JavaScript experiments.

The experiment studio joins ten staged courier investigations, four ECharts
instruments, four numerical mechanism labs, four step-through animations and three statistical investigations. The statistical companion connects nine original lessons to the supplied textbook chapters.
Controls change explicit models; the richer plots include values, assumptions,
accessible tables and CSV exports. Progress and notes are stored in the current
browser, with explicit import/export. There is no automatic device sync.

All 26 weeks have three diagnostic questions with choice-specific feedback,
repair explanations and matching workbook/code links. The revision queue mixes
missed, uncertain and due items; its transparent practice schedule is separate
from manual milestones. Text size, line width, section bookmarks, equation links
and diagram enlargement support longer reading sessions. Every reading links
to its GitHub source and a correction form that excludes personal progress.

Progress storage keeps the original schema-1 import format. A per-field journal
merges independent tab edits; note branches retain concurrent versions for user
review. Page hiding flushes pending changes. A visible status reports blocked
storage and exports remain available. See MDN’s [storage event](https://developer.mozilla.org/en-US/docs/Web/API/Window/storage_event)
and [visibilitychange](https://developer.mozilla.org/en-US/docs/Web/API/Document/visibilitychange_event)
for the browser lifecycle used here. Browser shutdown, storage eviction and
manual data clearing still make a separately saved backup necessary.

## Build and serve

Requirements: Node.js 22+, Python 3, Pandoc, `pdflatex`, `pdftocairo`, and the
course's existing TeX packages. Rebuild PDFs when their TeX sources change.

```bash
make portal
make portal-serve    # http://127.0.0.1:8767
```

Edit canonical content in `docs/`, the weekly folders, `research/` and `llm-lab/`.
`portal/scripts/prepare_content.py` snapshots these explicit roots, converts
lectures and figures, and builds the weekly ZIPs. It does not traverse the parent
workspace. `portal/scripts/build.mjs` resolves links and typesets mathematics;
invalid equations and unresolved source references fail the build.

`src/` owns the interface and chart controls. `generated/`, `dist/` and
`node_modules/` are disposable outputs. The public custom-domain release serves exactly the validated static `dist/` artifact through the existing hosting infrastructure. `study-rl.samuelzhang.co.uk` is canonical; `study-rl.samuelzhang.dev` redirects to it so browser-local learning progress uses one origin. Textbook PDFs are local source material and are excluded from website assets and weekly bundles.

## Verify

```bash
cd portal
npm run verify
npm test
node tests/browser.cjs /path/to/playwright-core http://127.0.0.1:9223 /path/to/@axe-core/playwright
node tests/learning-browser.cjs /path/to/playwright-core http://127.0.0.1:9223 /path/to/@axe-core/playwright
node tests/statistics-browser.cjs /path/to/playwright-core http://127.0.0.1:9223 /path/to/@axe-core/playwright
```

The browser suite uses an isolated context in a dedicated test browser and a
local portal server on port 8767. It checks lesson conversion, navigation,
notebook downloads, visual controls, search, progress backups, mobile layouts
and accessibility. Mathematical unit cases check the numerical models
independently of the plot output. See the [current learner review](../docs/LEARNER_REVIEW.md)
for the evidence and limits of the adversarial assessment.

## Executed notebook previews

Source notebooks remain output-free. A separate execution process records safe
outputs at `generated/notebook-outputs/manifest.json`, keyed by the repository
path of each notebook. Schema 1 entries record `source_sha256`, a `source_files_sha256` map for trusted course-code dependencies, `executed_at`,
`status`, `execution_mode`, any `overrides`, and cells with their original
one-based `index` and standard notebook `outputs`. Executed copies stay outside
the source repository. PNG figures, sanitised HTML tables and bounded plain text
are supported; scripts and SVG outputs are excluded. The build rejects results
whose exact notebook or declared helper bytes have changed, rejects symlink dependencies, and visibly distinguishes bounded runs. Publication verification requires current successful receipts for all 27 source notebooks.
No attached execution is labelled as source-only, with local execution guidance.

Preparation compares fingerprints before and after conversion. The build checks
that the prepared snapshot matches the current canonical materials. Release
verification then checks content, interface, execution-manifest and full-artifact
fingerprints. Run it against the exact artifact being published. Changes to
sources or generated outputs require another preparation/build/verification pass.

For visual development while other authors are still editing, explicit
`prepare_content.py --preview` / `build.mjs --preview` runs may produce a temporary
preview; these releases are marked and **cannot pass publication verification**.
A release requires stable preparation and a build without `--preview`.

If course functions move, run `node portal/scripts/refresh_handoffs.mjs` from the
repository root before preparing the release. It repairs source-line links by
verified Python symbols; ambiguous moved symbols require an authored correction.
The diagnostic tests continue to enforce exact source and notebook-cell targets.
