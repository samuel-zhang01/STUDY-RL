# Adversarial learner review

Reviewed on 2026-10-09 by a student-role agent, independently of the teacher
and curriculum-cohesion agents. This is an inspection of learning materials
and runnable examples, not a study of human learners. Scores describe the
evidence inspected; they do not establish how every student will perform.

## Acceptance rubric

The learner is comfortable with Python and the stated mathematical
prerequisites but new to reinforcement learning. Each dimension contributes
0–2 points; the total is out of 10.

| Dimension | Evidence required for full credit |
|---|---|
| Progression | One new difficulty at a time; named prerequisites; an explicit bridge from the previous idea |
| Explanation | Plain-language question, labelled quantities, a worked update, and a stated limit of the analogy |
| Visual causality | Changing a control changes a mathematically meaningful quantity; the display explains why; observations and expectations stay distinct |
| Exercise feedback | A prediction before the answer; an actual learner implementation; useful failure feedback; a reference run cannot masquerade as the learner's work |
| Transfer | A small runnable project, a changed assumption, an evaluation protocol, and a concrete connection to another week |

Acceptance requires **at least 8/10**, with **9/10** the target. A passing
average cannot excuse a mathematical error, a misleading success signal, or
an essential action that does not work. Revised materials will be scored only
after they are available for inspection.

## Baseline: useful explanations, insufficient interaction

The baseline review sampled the root README, study guide, weeks 1–2, DQN,
GAE, PPO, SAC, offline RL, RLHF, DPO, constraints, the synthetic audit, and
causal biology. It inspected the first-week notebook's exercise and test
cells. The worked arithmetic and caveats are a strength. A student still
needs a gentler entry point and feedback that proves what they actually did.

| Progression | Explanation | Visual causality | Exercise feedback | Transfer | Total |
|---:|---:|---:|---:|---:|---:|
| 1.2 | 1.7 | 0.6 | 1.3 | 1.5 | **6.3/10** |

These are the concrete demands sent to the teacher, cohesion agent, and
implementation agent:

1. **Give me a first experiment before a large installation.** I should be
   able to choose an action, see a reward, and update an estimate without
   installing PyTorch, Jupyter, SWIG, and Box2D first. Keep the full course
   environment available when its projects require it.
2. **Do not teach the whole course before week 1.** The study guide's shared
   delivery example includes Bellman equations, MC, TD, policy gradients,
   and termination. Tell me which subsection to read now and which to return
   to later. The same world should grow with my knowledge.
3. **Defer week 1's advanced material explicitly.** Beta posteriors, UCB
   regret bounds, LinUCB, and SNIPS appear beside first sample averages.
   Explain which are optional stretch work and link contextual/OPE ideas
   to week 11.
4. **Do not silently solve my coding exercise.** First-week scaffolds call
   `project.solutions`, while the notebook introduction describes unfinished
   functions. A green output can mean the reference worked. New practice
   files must require a learner edit, and instructions must distinguish
   learner and reference execution.
5. **Do not require a stochastic algorithm to win one trial.** The first-week
   contextual-bandit prompt says LinUCB has no per-run guarantee of winning,
   but its test asserts that it wins the selected trial. Use repeatable
   correctness checks and an honest multi-seed comparison for performance.
6. **Show the observation behind the bandit curve.** I want the chosen action,
   its count, the old estimate, the reward, and the new estimate. Explain why
   a lucky reward and low expected regret are different quantities. Let me
   explore before seeing the simulator's hidden means.
7. **Make future value move visibly.** Step a synchronous Bellman backup and
   highlight which old next-state value produced each new value. A smooth
   line by itself does not explain propagation. Label terminal reward and
   remaining value separately.
8. **Let me break the critic.** On the same recorded trajectory, change the
   next-state estimate and compare MC with TD. Then distinguish actual
   termination from a collector stopping early. For GAE, show which residual
   reaches the first advantage when lambda changes.
9. **Let the sign change the picture.** In PPO, move the ratio across both
   thresholds and flip the advantage. Mark the active branch and plateau;
   tell me the probability was not physically clamped. In DPO, keep reference
   and current likelihoods visible rather than hiding them behind one loss.
10. **Let unsupported data fail honestly.** Set a logger probability to zero
    while the target still chooses that action. Show that more samples or
    denominator clipping cannot recover its missing reward. Inspect weight
    concentration alongside the estimate.
11. **Show a budget's meaning.** A plot of mean cost should sit beside the
    possible cost of an individual episode. Let me see why an expectation
    budget permits occasional large costs, and how a shield changes the
    feasible action set.
12. **Give me a project ladder with feedback.** Start with a tiny delivery game
    or synthetic study scheduler, then add future consequences, a learned
    value, a constraint, and logs. Ask me to alter one assumption, inspect a
    failure, and evaluate on fresh seeds. Connect observational and
    intervention estimates before treating causal biology as a separate
    collection of acronyms.

## Revision review

### Guided coding/project path: 9.0/10

The revised walkthrough and standard-library projects meet the target for
their delivered scope: two-step delivery, the courier game, the synthetic
contextual study bandit, five learner functions, and a project ladder into
later weeks. This is a score for those learning entry points, not evidence
that the starter implements every advanced algorithm.

| Progression | Explanation | Visual causality | Exercise feedback | Transfer | Total |
|---:|---:|---:|---:|---:|---:|
| 1.9 | 1.9 | 1.5 | 1.9 | 1.8 | **9.0/10** |

The second pass independently ran all **21** standard-library project tests.
It also checked that unfinished exercise 1 prints `PENDING` and exits 1,
while the explicitly requested reference check passes the five supplied
answers and exits 0. The two-step CLI begins with values; target and policy
stages require an explicit choice. Hints open individually, chapter links
use the browser's actual titles, and the documented server serves the
repository root so lesson links remain reachable.

This pass requested and obtained repairs to the server instructions,
navigation labels, hint disclosure, and an overstatement that standardizing
observational rows was itself an intervention. The projects distinguish
fresh simulator evaluation from training evidence, and report hazard cost
separately even when a misleading reward encourages hazards.

### Browser learning lab: 9.1/10

| Progression | Explanation | Visual causality | Exercise feedback | Transfer | Total |
|---:|---:|---:|---:|---:|---:|
| 1.9 | 1.9 | 1.7 | 1.7 | 1.9 | **9.1/10** |

The first implementation did not pass unchanged. The learner review found
misleading exact-return gradient controls beside the PPO curve, a
slider/model probability mismatch, missing numerical bandit arithmetic,
a backup button that only highlighted already calculated values, and a
browser/Python map mismatch. These were repaired. The revised lab also
exposes per-action PPO terms, optional GAE residual weights and
contributions, possible per-trip costs, four Q-values for an inspected
grid cell, and the current greedy route as text. An additional hidden-panel
CSS bug was fixed before acceptance.

The final student pass used its own headless Chromium session, separately
from the implementation agent's browser. It inspected desktop and phone
screenshots and exercised all ten chapters at a 390-pixel viewport:

- Invalid seed feedback, a valid reset, and 100 actual bandit decisions.
- Bellman discount endpoints and the policy average.
- A next-state estimate of 8 followed by a terminal transition that removes
  the bootstrap.
- 100 courier training episodes, a learned goal route, and the terminal
  cell's four stored zero Q-values.
- An exact-return gradient step at the probability boundary, with matching
  model and slider values; GAE lambda 0 and 1; and signed PPO terms.
- An unsupported OPE target being withheld, followed by a supported logger
  and a working synthetic-log JSON download.
- A shield reducing expected cost from 0.5 to zero while displaying each
  individual trip's cost.
- Model reward error accumulating to 40 for horizon 40, discount 1, and
  per-step error 1, while the real extra reward stays zero.
- A length proxy preferring repetition and a separate bounded preference
  calculation.
- Standardization endpoints preserving the within-stratum route advantage.

The session recorded no JavaScript page errors, external requests, or page
overflow. Local weekly-guide and footer links inspected on the final
chapter returned HTTP 200. The default GAE control was actually hidden;
enabling it made its slider and contribution table visible. Desktop and
320-pixel phone screenshots were inspected for reading order and chart
legibility. Broader cross-browser/layout testing is recorded separately by
the implementation agent.

The final axis refinement was checked again in Chromium: success-rate ticks
are 0, 0.5, and 1; the PPO curve uses its actual range and visibly flattens;
the surrogate at the old policy displays zero rather than rounded negative
zero. The plot remains readable at a 390-pixel viewport.

All **12** final engine tests passed independently, including invalid and
nonfinite input guards. A separate 30-seed probe
on the final canonical map trained 100 episodes per seed; all 30 learned a
six-move greedy goal route. Exploratory training returns and hazard counts
varied. The GAE implementation also agreed with an independent backward
calculation at lambda 0, 0.5, and 1. These are measured fixture results,
not general learning or safety guarantees.

The remaining gap from 10 is educational: prediction questions reveal
reasoning but do not grade a learner's written explanation, and some
advanced mechanisms still rely on their weekly calculation and code
rather than a dedicated browser simulation.

### Connected curriculum: 8.8/10, every track at least 8

All 26 new weekly bridges were inspected, alongside the worked examples
and code connections in their guides. Their arithmetic, prerequisite
connections, and scope statements are sound after correcting route/desk
terminology, directing proxy failures to the preference experiment, and
distinguishing the static shield fixture from the Python grid. New early
notebook introductions now state that supplied wrappers call references;
they no longer claim that execution is an independent learner attempt.

The table scores the **ease of entering and connecting each track**, using
the revised path, guides, browser mechanisms, and exercise directions. It
does not score a beginner's mastery of every paper, establish benchmark
performance, or imply that each named algorithm is implemented in the lab.
The unweighted mean of these track scores rounds to 8.8.

| Track | Weeks | Score | Evidence and remaining boundary |
|---|---|---:|---|
| Foundations | 1–4 | **9.3** | One action/reward, then delayed reward, sampled return, bootstrap, and a real learned route; first/every-visit MC and SARSA remain weekly calculations. |
| Deep value learning | 5–6 | **8.5** | Explicit table-to-network bridge, labelled batch/terminal targets, Double selection/evaluation and replay counterexamples; the browser has no neural trainer. |
| Policy methods | 7–10 | **9.0** | Same delivery values, direct probability updates, GAE weights, signed PPO terms; continuous density/Jacobian and SAC/TD3 use their weekly exercises. |
| Context, OPE and matching | 11–12 | **8.9** | Genuine contextual starter, actual logged propensities, support failure, weight inspection and examination correction; ranking/matching is broader than the two-route toy. |
| Multi-agent learning | 13 | **8.5** | A tiny board and explicit player-perspective backup make sign changes inspectable; self-play/search stays in the weekly project. |
| Offline and sequence policies | 14–15 | **8.4** | Missing-support counterexample, worked expectile/behavior-frequency calculations, return conditioning and attention-mask example; no browser IQL/DT/diffusion trainer. |
| Model-based learning | 16 | **9.0** | Manipulable imagined/real reward gap and horizon; the guide distinguishes transition error and latent-model training from this reward-bias fixture. |
| Exploration and adaptation | 17 | **8.3** | Exploration carries forward; worked novelty decay and one-step adaptation expose the new objectives; RND/MAML have no dedicated interactive browser experiment. |
| Language-model RL and preferences | 18–21 | **8.8** | Policy-to-token/reference bridge, preference margin, proxy failure and the practical LLM lab; the two-answer loss is not PPO/GRPO training or quality evaluation. |
| Safety and capstone | 22–24 | **9.1** | Visible reward/cost frontier and individual outcomes, executable hazard-reward failure and shield, then separate evaluation/constraint claims. |
| Applied synthetic audit | 25 | **8.8** | Reuses propensities, support and separate metrics to produce a bounded audit memo; simulator evidence remains scoped to a synthetic shortlist. |
| Causal extension | 26 | **8.8** | Same-mixture comparison, numerical confounding reversal and explicit assumptions; standardization is distinguished from intervention and general graph discovery. |

The delivered learning entry points meet the requested **8/10 minimum**;
the guided coding path and browser lab meet the **9/10 target**. Useful next
extensions are a replay/target-network visualizer, a novelty-versus-task
dashboard, and a stepwise search tree. They would improve the advanced
tracks without adding more material to the first session.
