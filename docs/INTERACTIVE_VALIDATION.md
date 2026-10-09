# Interactive learning validation

Checked on **9 October 2026**. The revision adds ten browser experiments, two
standard-library coding projects plus a shared numerical anchor, five learner
functions, and prerequisite/transfer bridges across all 26 weeks. The
[student review](LEARNER_REVIEW.md) records the adversarial findings and rubric:
coding projects **9.0/10**, browser lab **9.1/10**, connected curriculum
**8.8/10**, and individual tracks **8.3–9.3/10**. These are agent review scores,
not results from a human learning study.

## Checks actually performed

| Check | Evidence |
|---|---|
| Existing course and LLM suites | 429 pytest checks pass, including the new HTML/source release safeguard |
| New Python projects | 21 standard-library tests pass; terminal/cutoff semantics, shielding, logged support, deterministic fixtures and CLI error behavior are checked |
| Browser numerical models | 12 Node tests pass; independent hand targets, signed PPO terms, GAE endpoints, OPE support/weights, causal reversal, invalid inputs and multi-seed learning are checked |
| Adversarial learner | Independently ran Python/engine checks, a 30-seed courier probe and actual interactions in all ten Chromium chapters |
| Browser workflows | Chromium and WebKit pass locally; fresh Linux Chromium, Firefox and WebKit each pass 330 assertions across all ten screens at 320, 390, 768 and 1280 pixels |
| Interaction boundaries | Invalid seeds preserve observations; reset clears them; pause/navigation stop animations; reduced motion advances one step; keyboard sliders change displayed results |
| Logs and navigation | Exported JSON matches the visible seed/probabilities/200-row dataset; all chapter/footer course links return HTTP 200; no third-party requests or page exceptions |
| Visual inspection | Desktop courier game and narrow PPO/GAE layout inspected; probability axes remain within 0–1, PPO uses its actual objective range, optional GAE stays hidden initially |
| Notebook honesty | First 25 introductions identify reference wrappers; week 26 retains its explicit learner/reference selector; week 1 executes all 36 cells in a fresh kernel |
| Source hygiene | All 27 notebook schemas/code parse and have no saved outputs; local documentation/assets resolve; whitespace and dependency checks pass |
| Privacy | Gitleaks 8.30.1 finds no credentials in reachable history or the public-source candidate tree; new material uses synthetic examples |

There are **462 numerical/code checks** in the complete local test command
(429 pytest + 21 unittest + 12 Node). Browser assertions are reported separately
because they exercise the same workflows at multiple sizes; they are not 330
independent educational claims. Tests establish these observations, not proof
that all possible behavior is flawless.

## Compatibility and scope

Local checks used macOS 27.0.1 on Apple Silicon, Python 3.12.15, Node 20.19.6,
and the locked Playwright 1.64.0 development dependency. The Firefox test binary
could not launch its temporary profile on this Mac. Chromium and WebKit did
launch and passed. The fresh Ubuntu run passes all three browsers, including Firefox: **330
assertions per browser, 990 total**. Its CPU job also passes all 462 code checks. Browser tooling is only used for
development; the lab itself has no external runtime packages or requests.

The grid is deliberately small and deterministic. A six-move learned route on
its fixtures is not a guarantee for noisy or larger games. MC/TD isolates a
bootstrap error; it does not empirically estimate the full variance trade-off.
The policy, GAE, model-bias and DPO displays isolate exact mechanisms. The
weekly code and readings remain necessary for full Rainbow, continuous-control,
self-play, diffusion, world-model and language-model methods. A perfect toy
shield and observational standardization do not establish deployment safety
or causal identification.

## Fresh Linux CI

At code revision `2459771`, [course and browser checks](https://github.com/samuel-zhang01/STUDY-RL/actions/runs/37979911086)
pass in a fresh Ubuntu environment, and the [history secret scan](https://github.com/samuel-zhang01/STUDY-RL/actions/runs/37979911129)
passes. The first Linux browser run exposed a WebKit tablet-width control
wrap that the Mac checks missed. Labels now wrap consistently across fonts,
and the final three-browser run passes all viewport checks. Changing a
control also pauses animation and reports the actual paused state.

## Reproduce

From the repository root, install the course/lab dependencies as documented in
the [README](../README.md), and use Node.js for the development checks:

```bash
make tests PY=python
make release-check PY=python
cd interactive
npm ci --ignore-scripts
npx playwright install chromium firefox webkit
npm run test:browser
```

For a Mac where the Firefox test binary cannot launch, set
`RL_LAB_BROWSERS=chromium,webkit` for the supported local checks and retain the
three-browser Linux CI run. Source logs/screenshots and executed notebooks
remain ignored local verification artifacts. The public repository is still
private pending removal of the original GitHub cached content described in
the [release review](RELEASE_REVIEW.md).
