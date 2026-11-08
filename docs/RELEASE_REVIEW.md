# Current release and maintenance notes

The public course is available at **https://study-rl.samuelzhang.co.uk/**.
[The .dev address](https://study-rl.samuelzhang.dev/) redirects to the same
canonical origin so existing browser-local progress remains accessible.
[GitHub](https://github.com/samuel-zhang01/STUDY-RL) holds the source and downloadable
notebooks. The October 20 portal revision is based on the existing sanitized
public history; it does not merge the former private checkout's history.

## What belongs where

- The website is the student entry point: instructions, full lessons,
  experiments, notebook previews, weekly downloads and local progress.
- `docs/` holds maintained course navigation, notation and the current review.
- `week-*/` holds canonical teaching guides, TeX/PDFs, notebooks, projects and tests.
- `interactive/` holds the ten staged courier experiments. `study-lab/` holds
  four smaller calculation labs used by the visual atlas.
- `learning_projects/` is the standard-library coding path with explicit TODOs
  and independent feedback. The website has no Python runtime.
- `portal/` builds the unified website and its downloads. Generated content,
  distributions and dependencies are ignored in this source repository.
- `docs/archive/` preserves dated reviews as historical evidence; these are not
  current test results and are excluded from the main lesson search.

Week 25 remains `week-25-synthetic-ats-rl-audit`; the canonical syllabus is
`syllabus/syllabus.pdf`. Notebook outputs, model weights, private settings and
credentials remain excluded. The release checker validates source hygiene,
notebook structure and local links; portal checks additionally validate the
rendered course and student workflows.

## Reproduce validation

```bash
make release-check
make course-tests
make learning-tests lab-tests
make portal
make portal-serve
```

For a complete new publication, `make portal-release` first executes all 27
reference notebooks in isolated copied trees, then prepares and builds the
website, tests it, and verifies source/output/artifact fingerprints. The
source notebooks stay output-free. The generated execution manifest supplies
the website's static figures, result tables, recorded settings, and provenance;
learner code still runs locally in Jupyter. Use `make portal-verify` immediately
before publishing the exact artifact. A retained `dist/` or an old successful
notebook run cannot certify changed source code.

The [November upgrade plan](UPGRADE_PLAN.md) defines the 9.8/10 teaching and
9.5/10 visual review thresholds. Its [textbook validation record](statistics/VALIDATION.md)
states which probability/statistics sources and numerical cases were checked.
The [November upgrade review](UPGRADE_REVIEW.md) records the current material,
notebook and website checks. The October learner review remains dated
historical evidence.

See [portal development](../portal/README.md) for the browser/accessibility suite,
[the current adversarial learner review](LEARNER_REVIEW.md) for acceptance evidence,
and [historical records](archive/README.md) for earlier runs. A fresh portal
snapshot must follow any content/PDF change. Weekly ZIPs preserve the source
folder structure and include shared helpers; their notebook copies are checked
against the canonical notebooks before publication.

The first four weeks have a smaller `requirements-foundations.txt` environment.
The full environment includes PyTorch and Gymnasium's Box2D dependencies; the
pretrained LLM lab remains separately installed. Reference notebook execution
is evidence that an example runs, not evidence that a learner wrote it.

This review does not certify all bibliography entries, reproduce every paper,
benchmark GPU training, or establish clinical/production safety. The earlier
[public-release history review](archive/RELEASE_REVIEW_2026-10-06.md) is retained
with its original dated scope; statements about repository visibility there
are historical.
