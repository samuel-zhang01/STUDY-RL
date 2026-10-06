# Teaching review of STUDY-RL

The curriculum has substantial breadth, useful applied threads, and runnable
toy implementations. Its main teaching weaknesses were uneven worked examples,
compressed derivations, and statements that made a heuristic or a simplified
implementation sound more general than the evidence supported.

The earlier teaching revision added a teaching guide to every existing week, with 64 numbered worked
examples and 75 hidden-answer self-checks, connects them through
[STUDY_GUIDE.md](../STUDY_GUIDE.md), and corrects specific mathematical and
explanatory errors in the existing materials. It is a teaching and targeted
correctness review; it is not a reproduction of every cited paper or a fresh
audit of an external production system.

## What changed for the learner

| Before | Enhancement | Learning purpose |
|---|---|---|
| Many algorithms arrived as dense notation | A central derivation per teaching guide, with symbols and assumptions stated | Explain what permits each step |
| Worked examples varied in depth | At least two small calculated examples per week, including a boundary case | Predict, calculate, then interpret |
| Analogies sometimes implied more than the mathematics | Explicit analogy mappings and limits | Use intuition without inheriting a false guarantee |
| Formula, notebook, and implementation could tell different stories | Links to actual code and explicit simplifications | Trace an equation into the implementation that exists |
| Running reference solutions could look like mastery | Hidden-answer retrieval and transfer questions; revised progress milestones | Separate demonstration from independent understanding |
| Related algorithms used different notation and examples | Reward-index translation and a shared two-step problem | Compare algorithms while holding the problem fixed |

Each workbook links to its local `teaching_guide.md`. The longer lectures and
reading lists remain available for depth; the guides provide an entry point and
a way to check understanding before moving on.

## Important corrections

The individual guides give the calculations and primary-source links. These
are representative corrections, grouped by the misunderstanding they prevent.

| Area | Correction and implication |
|---|---|
| Bandits | Compare regret with regret, not regret with a bound on pull counts. UCB confidence bonuses, posterior sampling, and epsilon exploration have different mechanisms. |
| Dynamic programming | A changed greedy action does not imply strict value improvement when actions tie. In-place backups and synchronous backups use different intermediate values. |
| Monte Carlo | A backward scan that accepts the first state it encounters selects the last chronological visit. Importance ratios for a deterministic target can be very large on matching trajectories. |
| Function approximation | Tabular contraction and convergence statements do not automatically transfer to a nonlinear approximator. The deadly triad is a source of possible instability, not a theorem that every instance diverges. |
| GAE and TRPO | A full terminal trajectory and a truncated rollout have different bootstrap boundaries. A practical average-KL constraint is distinct from a maximum-state divergence assumption in a bound. |
| PPO | The negative-advantage branch is flat below the lower ratio threshold. Clipping changes incentives; it does not impose a hard KL or parameter-step bound. |
| SAC | A density integrates to one; its log density does not integrate to zero. The squashing Jacobian and the distinction between a loss gradient and a descent step determine the correct sign. |
| Off-policy evaluation | Support, nuisance-model fitting, and evaluation selection matter. A perfect conditional reward mean does not eliminate random outcome residuals in DR. SNIPS is not universally lower-variance or unbiased. |
| Ranking | Position exposure, conditional selection rates, and shares among selected candidates are different quantities. Metric names must match their denominators. |
| Multi-agent search | Value signs must consistently use either the player to move or a fixed player's perspective. A sign error in the toy MCTS backup was corrected. |
| Offline and model-based RL | Conservative objectives and return conditioning need assumptions; neither gives an unrestricted deployment guarantee. Learned-model planning introduces model error as well as sampling error. |
| Exploration | The pseudo-count formula needs the pre-update and post-update density in its denominator. RND prediction error is not guaranteed to fall everywhere after an SGD step. |
| RLHF | A sampled log probability ratio can be negative although the corresponding distribution-level KL is nonnegative. Correct critic means do not make all TD errors deterministic. |
| Preference optimisation | Keep the reward reparameterisation and preference-temperature convention consistent; inserting the temperature twice changes the DPO objective. |
| Group baselines | Baseline cancellation requires the appropriate independence or conditional-expectation argument. A zero-sum batch of advantages does not imply a zero policy gradient. |
| Oversight | A KL anchor controls policy deviation; it does not identify an arbitrary additive constant in reward-model scores. |
| Constraints and shields | With reward-maximisation Lagrangian $L=J_r-\lambda(J_c-d)$, the policy maximises and the multiplier minimises. Expected cost constraints, a heuristic penalty, and model-based per-action shielding are different assurances. |
| Application audits | Synthetic demonstrations and same-log estimates need to be labelled as such. They are not held-out evidence about a production recruitment system. |

## Improvements to the learning workflow

- [GLOSSARY.md](../GLOSSARY.md) translates reward timing and distinguishes return
  from value, online from on-policy, and sampled residuals from expectations.
- [PROGRESS.md](../PROGRESS.md) asks for explanation, hand calculation, transfer,
  and evaluation, with a short error-log template.
- [WEEK_BUNDLE_SPEC.md](../WEEK_BUNDLE_SPEC.md) now specifies the teaching-guide
  structure and meaningful learner checks. It identifies requirements as
  authoring targets instead of asserting universal compliance.
- [scripts/new_week.py](../../scripts/new_week.py) scaffolds a teaching guide and links
  it from the new workbook, so future weeks retain the improved entry point.
- The consolidated notebook hints now reuse exercise-specific inline hints
  behind collapsible answers, replacing the same generic advice repeated for
  every exercise. `scripts/append_hints_section.py --refresh` rebuilds only the
  generated section and leaves learner code in place.
- `make tests` runs `common` and every week in separate processes. Previously
  the pytest configuration's default paths covered only some weeks.
- `make notebooks` explicitly uses each week's working directory, matching the
  notebooks' local `project` imports.
- The QA inventory's documentation now describes presence checks accurately;
  its test sweep uses the pytest exit status rather than finding the word
  “passed” in output that might also report failures.

## Remaining work worth prioritising

These limitations belong to the existing course and are documented in the
relevant guides. They should guide a subsequent implementation or research
reproduction pass.

1. **Make learner assessment independent throughout.** Many workbooks either
   delegate directly to the solution module or automatically fall back to it.
   The new transfer questions help now, but a uniform learner/reference mode
   and checks against independently written student functions would make the
   course a more reliable assessment tool.
2. **Extend rollout checks when adding environments.** The October code review
   repairs termination/truncation and vector-autoreset handling in the affected
   trainers, and preserves Gaussian samples for consistent likelihood ratios.
   New wrappers, finite-horizon objectives, and continuous-action policies still
   need their own checks before reusing these teaching recipes.
3. **Strengthen application evaluation.** Separate training, policy selection,
   nuisance-model fitting, and final evaluation. Report overlap and uncertainty,
   and distinguish policy action probabilities from calibrated outcome
   probabilities. Rolling constraints may require history in the state.
4. **Reproduce larger claims explicitly.** Some syllabus experiments describe
   larger environments and models than the bundled projects implement. Record
   experiment configs, actual runs, baselines, seeds, and uncertainty before
   presenting a toy result as a paper reproduction.
5. **Verify the research dossiers and extended-reading quotations separately.**
   Primary sources were checked for the corrections in this pass. Every
   bibliography entry, quoted passage, and time-sensitive “state of the art”
   claim across the reading archive has not been independently re-audited.

## Validation of the earlier teaching revision

Results for each week are in [teaching_validation.tsv](teaching_validation.tsv).
Validation used a local Python 3.12.14 virtual environment, PyTorch 2.14.0,
Gymnasium 1.3.0 with Box2D, NumPy 2.5.2, and TeX Live 2024.

- **All 25 workbooks executed successfully** in fresh kernels with each week
  as the working directory. After the final Markdown and hint edits, every
  executable cell was compared with its successful execution; code was
  unchanged. All final notebooks pass `nbformat.validate`, with missing cell
  IDs added and existing IDs, code, and outputs preserved.
- **279 tests passed; one test failed.** The common suite and all weeks ran
  in separate processes. The failure is the existing week-25 assertion
  `test_train_runs_and_lints_competitive`: LinTS SNIPS was `0.6237545`, while
  uniform SNIPS was `0.6773098`; the test requires LinTS to be at least
  `0.6273098`. The week-25 runtime and tests were unchanged. This is an
  empirical-ordering assertion, not an algorithmic guarantee; it was retained
  and reported, rather than weakened. In that recorded environment,
  `make tests` exited nonzero at this assertion. The later code review below
  supersedes this historical test status.
- **All 25 lecture PDFs and the syllabus rebuilt.** The lecture PDFs contain
  448 pages. Every lecture page was rasterised for a page-edge check; no edge
  cutoffs were detected, and no lecture log reports an overfull box above 5 pt.
  Representative revised equations, figures, learning objectives, and exercises
  were visually inspected. Long exercise labels, an overwide equation and
  diagram, and overlapping week-22 sidenotes were reformatted. These checks
  do not constitute a complete visual review of every paragraph or of the
  unchanged extended-reading PDFs.
- **Numerical examples were checked independently**, including exact sums,
  probabilities, importance weights, gradient signs, GAE boundaries, baseline
  variances, and constraint updates. The shared example's analytic gradient
  also agrees with a finite-difference calculation. Selected examples were
  cross-checked against actual solution helpers and PyTorch autograd.
- **All local links in the 25 guides and revised navigation documents resolve.**
  The hint generator was checked for separate-cell hints, preservation of code,
  and idempotent refresh; the week generator produced a schema-valid notebook
  in a temporary directory. `git diff --check` passes.

The historical test command was:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m pytest week-25-synthetic-ats-rl-audit/tests -q
```

To rerun a workbook, substituting the desired week directory:

```bash
source .venv/bin/activate
papermill --kernel python3 --cwd week-01-mdps-bandits \
  week-01-mdps-bandits/workbook.ipynb \
  week-01-mdps-bandits/workbook.executed.ipynb
```

The extended-reading archive and large benchmark experiments were not rebuilt
or reproduced as part of this teaching pass.

## Causal-biology extension and code review — 5 October 2026

The new [Week 26 guide](../../week-26-causal-discovery-biological-data/teaching_guide.md)
contains a five-session, ten-hour plan, three calculated examples, eight workbook
exercises, and a synthetic patient-data experiment. The overlap review found that
weeks 11/12/14/25 already cover logging, support, and evaluation; they do not
teach biological multi-environment discovery, observational equivalence, or how
measurement enhancement can change dependencies. The extension focuses on those
questions and links primary sources in [its dossier](../../research/causal_biology.md).

The source review verified the public event listing, effect-identification
readings, multi-environment S-FCI, invariant causal prediction, and published
task-driven augmentation and imputation studies. The lesson distinguishes these
sources from Synlico's proprietary platform descriptions. Its residual diagnostic
and nuisance randomization do not implement PC, FCI, S-FCI, or ICP. Predictions
and adjusted effects are separate experiments, with no drug-target validation
claim.

Parallel reviews covered shared utilities, all 25 existing weekly projects and
their workbook/code connections, the practical LLM lab, and maintenance scripts.
The repairs include:

| Area | Repair and teaching implication |
|---|---|
| Shared utilities and foundations | Match terminal environment transitions to the DP model, use reproducible bandit seeds, evaluate the policy actually returned by a budget-limited iteration, and preserve car counts when transfers meet capacity. |
| TD and deep value learning | Bootstrap through collection time limits, fix the off-policy action ratios in Baird's fixture, and disable NoisyNet perturbations for deterministic evaluation. |
| Policy methods | Preserve sampled Gaussian actions for likelihood ratios while clipping actions at the environment boundary; bootstrap from final observations and stop GAE traces at resets; make PPO ablation flags affect their losses. |
| Bandits, ranking, and search | Log the full exploration/exploitation mixture probability, use one reward mechanism for logs and oracle comparisons, label calibration and training objectives accurately, and sample search noise over legal moves. |
| Offline and sequence learning | Keep episode boundaries separate from true terminals, average repeated tabular minibatch updates, weight IQL policy extraction by logged action counts, and mask DT padding. |
| Model-based and meta learning | Track actual prior actions in recurrent state inference and distinguish full-information meta-bandit gradients from sampled exploration. |
| LLM toy trainers | Match sampling and recomputation temperatures, control the frozen-reference KL separately from PPO's update KL, and label fresh arithmetic prompts accurately when training exposure can overlap. |
| Safety and application audits | Correct the pendulum equilibrium sign, reject an empty shield safe-action set, preserve the offline checkpoint during finetuning, use an empirical differentiable fairness penalty, and evaluate fixed ATS policies on fresh logs. |
| Build and validation tools | Run weekly pytest suites in separate processes, reject mixed-week collection, validate new-week arguments, and fail when required artifacts or PDF rendering checks are missing. |

The syllabus now names the actual bundled mini-projects, including GridWorld,
CartPole, character models, and LavaGridWorld. Paper topics and larger benchmark
experiments are identified separately. A2C/PPO plot generators no longer extend
an episode curve with invented repeated tail values, and ablation titles do not
announce a winner before measuring one. The week-9 reading was corrected against
primary sources to remove invented quotations and unsupported claims.

On the new causal fixture, seed 0 gives naive ATE 2.2106 and adjusted ATE 1.0009
against a known effect of 1. Across seeds 0–4, mean adjusted ATE is 1.0006 (sample
SD 0.0032). Held-out prediction MSE averages 5.2523 for raw ridge, 2.2166 for
validation-selected enhancement, and 1.7898 for the no-proxy ablation. These are
synthetic run outcomes; the SD measures variation across those five fixtures.
The ablation can outperform the selected enhancement, illustrating that source
validation cannot guarantee the best choice for an unseen shift.

Full paper reproductions, clinical efficacy, and a complete verification of
every historical bibliography entry remain outside the evidence from this
review. Existing large-experiment figures that were not regenerated remain
historical illustrations, not new benchmark runs.

### Validation of the current revision

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg make qa` completed successfully.
The isolated suites passed 425 tests: 329 weekly tests, 9 shared-utility tests,
4 maintenance-script tests, and 83 practical LLM-lab tests. This count comes from
the actual per-process results, rather than combined collection with aliased
test modules. Mixed-week collection was separately checked to fail with an
actionable message.

All 26 weekly workbooks executed in fresh kernels; their saved executions match
the current cell sources and contain no error outputs. The numerical LLM-lab
workbook also passed `make llm-notebook`, without downloading model weights.
Python syntax checks passed for 308 files and 413 notebook code cells, and all
27 notebook schemas are valid. The guides now contain 67 numbered worked
examples and 80 hidden-answer self-checks.

All 52 weekly lecture/reading PDFs and the syllabus compile. The final raster
and compile-log sweep reports no edge cutoffs or overfull boxes of at least
5 points. The three workbooks and week-24 lecture changed during the last
explanation pass were refreshed before checking the final artifacts.
See [per-week teaching results](teaching_validation.tsv),
[artifact results](qa_report.tsv), and [PDF layout results](pdf_margin_report.tsv).
Local Markdown links resolve and `git diff --check` passes.

Validation used Python 3.13.9, NumPy 2.4.6, PyTorch 2.14.0 on CPU,
Gymnasium 1.4.0, Papermill 2.7.0, and TeX Live 2024. Non-failing dependency
deprecations, tensor-to-scalar logging warnings, and a PEFT configuration warning
remain visible in the test output. GPU training and full benchmark reproduction
were not part of these checks.
