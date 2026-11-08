# Attribution and source material

STUDY-RL contains original explanations, exercises, synthetic fixtures, and
reference implementations. Its history credits Samuel Zhang
(`samuel-zhang01`) and Guglielmo Cimolai for its development.

The [MIT license](../LICENSE) covers contributions for which the contributors
hold the necessary rights. Cited books, papers, websites, external libraries,
pretrained weights, separately downloaded datasets, and trademarks retain
their own licenses and terms.

## Textbook references

The probability and statistics lessons use Kyle Siegrist's *Probability,
Mathematical Statistics, and Stochastic Processes* and Sheldon M. Ross's
*Introduction to Probability and Statistics for Engineers and Scientists*,
fifth edition, as methodological references. The source catalog in the
statistics lessons records the sections used. Course explanations and numerical
examples are original; a citation does not relicense a source.

The locally supplied book PDFs and the instructor manual included in the Ross
file stay outside Git, website assets, and weekly downloads. Book pagination
is checked section by section because the supplied files include front matter
and additional material. Link to authorized source pages rather than copying
book chapters or instructor solutions into the course.

## Models and teaching data

The optional lab defaults to `HuggingFaceTB/SmolLM2-135M-Instruct`. Consult its
model card before downloading or redistributing weights. Downloaded weights
and generated adapters are excluded from the repository.

Insurance, recruitment, and patient examples use locally generated synthetic
data. The ATS architecture is a hypothetical teaching design. It does not
describe or audit a private production system. Research and event references
for the biology extension appear in its [source dossier](../research/causal_biology.md).

## Website dependencies

The website bundles Apache ECharts (Apache-2.0), Mermaid (MIT), and KaTeX (MIT).
Their license files, ECharts NOTICE, and exact versions ship under `vendor/`
in the built website. Build-time Markdown parsing and sanitization dependencies
are recorded in [the portal lockfile](../portal/package-lock.json).
