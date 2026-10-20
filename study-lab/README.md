# Interactive RL experiments

[Choose a route](../docs/LEARNING_MAP.md) · [Visual explanations](../docs/CONCEPT_ATLAS.md) · [Coding walkthrough](../docs/INTERACTIVE_LEARNING.md)

Open **[index.html](index.html)** in a browser from your local checkout. It works
offline with no packages, account, downloads or build step. GitHub displays HTML
source rather than running it: download/clone the repository, then open
`study-lab/index.html` from your file manager.

Alternatively, from the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Open **http://127.0.0.1:8765/study-lab/**. Stop the server with Ctrl+C.
Keep the adjacent `model.js`, `app.js` and `style.css` files together.

| Experiment | Use after | Predict before changing the controls |
|---|---|---|
| Return and advantage | Weeks 1–2, revisit in 7 | At what discount does continuing become better? |
| TD boundaries | Weeks 3–4, revisit in 8 | Does stopping collection remove the future from the task? |
| PPO clipping | Week 9 | Which sign makes the lower threshold a plateau? |
| Logged support | Week 11, revisit in 14 | Can an estimator recover an action the logger never tried? |

Every experiment includes visible arithmetic, a diagram or comparison, a
prediction check with explanation, a reset, and a link back to its lesson.
Use Tab to reach controls and arrow keys to adjust sliders. Answers and reading
position are not saved; record learning evidence in [PROGRESS.md](../docs/PROGRESS.md).
The lab's checks are short retrieval prompts, not a mastery certificate.

The calculations are deterministic teaching examples, not training runs.
The support experiment deliberately keeps its finite log fixed as you change
the logging probability, and labels impossible or unsupported configurations.
It does not estimate confidence intervals.

Without JavaScript, use the worked examples in the
[visual explanations](../docs/CONCEPT_ATLAS.md). To verify the lab arithmetic and
model checks, run `make study-check` with Python, pytest and Node.js 18+
installed. Those tools are for maintenance; opening the lab needs only a browser.
Browser checks are documented in the
[student-experience audit](../docs/LEARNER_REVIEW.md).
