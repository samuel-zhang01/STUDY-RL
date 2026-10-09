# The courier learning lab

Ten small experiments connect chosen-action feedback, Bellman values, MC/TD,
Q-learning, policy gradients/PPO/GAE, logged-policy evaluation, safety, model
error, preferences and confounding. Use the
[progressive activity guide](../docs/INTERACTIVE_LEARNING.md), then the
[26-week learning path](../docs/LEARNING_PATH.md). Each screen includes a
prediction, hidden reasoning, live calculations and a transfer task.

From the **repository root**, run:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Visit `http://localhost:8000/interactive/`. Serving the repository root keeps
the weekly guide and project links working. `make lab PY=python` does the same
thing. Ctrl+C stops the server. You can also open `index.html` directly; the
simulations do not fetch files or need internet access. Your editor can open the
linked Markdown guides. No state or personal data is sent anywhere.

Begin with **1 · Try a route**. The route means are hidden until you request
them. Later screens reuse the delivery decision or add a larger grid game.
Changing experiments resets that experiment and stops any animation. A seed
reproduces the same sequence **when you repeat the same choices and settings**;
different policies consume randomness differently. This is not a paired-seed
benchmark. The graphs come from actual simulation observations or explicitly
named exact formulas, never prewritten training curves.

The game uses zero-based `(column,row)` coordinates. Start is `(0,0)`, goal is
`(3,3)`, and hazards are `(1,2)` and `(2,2)`. A goal/hazard ends the task; a
60-step collection cutoff retains a bootstrap. The Python project uses the
same world with randomized tie-breaking and its own random generator, so seeds
do not produce identical trajectories across languages. A perfect static
shield here illustrates a mechanism, not real-world safety certification.

Controls work with keyboard input. Animations can pause, cancel on lesson
changes, and become one-step advances when reduced motion is preferred. Charts
resize to narrow screens and have named axes; exact arithmetic and tables
provide text alternatives. Prediction answers are never treated as a grade.

## Development checks

The lab has **zero runtime dependencies**. Node.js is only needed for development
tests, and Playwright is only needed for browser checks:

```bash
node --test interactive/tests/engine.test.js
python -m unittest discover -s learning_projects/tests -v
cd interactive
npm ci --ignore-scripts
npx playwright install chromium firefox webkit
npm run test:browser
```

The browser runner starts and stops its own loopback server. It tests all ten
screens at 320, 390, 768 and 1280 pixels; numeric updates, animation cancellation,
seed validation, keyboard input, reduced motion, dark appearance, exports,
course links and absence of external requests. Set `RL_LAB_BROWSERS=chromium`
to run only Chromium, or `RL_LAB_TEST_PORT=8878` if the default port is occupied.
Browser installation follows the [official Playwright library instructions](https://playwright.dev/docs/library).

`engine.js` contains the numerical models; `app.js` contains the explanations,
controls and SVG graphs. The Python games deliberately keep the learner TODOs
separate from reference implementations. Their checker exits unsuccessfully
while an exercise is pending, as documented in the activity guide.
