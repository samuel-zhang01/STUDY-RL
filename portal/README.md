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
instruments, four numerical mechanism labs and four step-through animations.
Controls change explicit models; the richer plots include values, assumptions,
accessible tables and CSV exports. Progress and notes are stored in the current
browser, with explicit import/export. There is no automatic device sync.

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
`node_modules/` are disposable outputs. A separate Sites source snapshot includes
`generated/` so it can rebuild without TeX. The existing private Sites project
is recorded in `.openai/hosting.json`; the public custom-domain release serves
exactly the validated static `dist/` artifact.

## Verify

```bash
cd portal
npm test
node tests/browser.cjs /path/to/playwright-core http://127.0.0.1:9223 /path/to/@axe-core/playwright
```

The browser suite uses an isolated context in a dedicated test browser and a
local portal server on port 8767. It checks lesson conversion, navigation,
notebook downloads, visual controls, search, progress backups, mobile layouts
and accessibility. Mathematical unit cases check the numerical models
independently of the plot output. See the [current learner review](../docs/LEARNER_REVIEW.md)
for the evidence and limits of the adversarial assessment.
