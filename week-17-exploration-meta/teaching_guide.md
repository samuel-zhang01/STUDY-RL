# Week 17 teaching guide: novelty and learning to adapt

Read this beside [the notes](notes.tex), then attempt [the workbook](workbook.ipynb).
The numerical examples below are constructed teaching cases, not training results.

## Learning bridge

Follow [the staged learning path](../docs/LEARNING_PATH.md) when a topic is new.

- **Carry forward:** Retrieve week 1’s exploration–exploitation choice and [week 16’s](../week-16-model-based/teaching_guide.md) model uncertainty.
- **Add one difficulty:** Novelty can encourage visiting an unfamiliar place, while meta-learning adapts across related tasks. A novelty score and task success measure different things.
- **Try after the worked example:** Use [the RND exercises](project/solutions.py). Compare a known room with extrinsic reward 1 and bonus 0 to a novel room with reward 0 and bonus 2, weighting bonus by 1. Which wins the training objective; which wins the original task? The browser bandit illustrates trying uncertain choices, not RND or MAML.

<details><summary>Check your prediction</summary>

The novel room wins the augmented objective, 2 versus 1. The known room wins the extrinsic task, 1 versus 0. Report both rather than calling novelty alone success.

</details>

**Carry it onward:** [RLHF](../week-18-rlhf-ppo-llm/teaching_guide.md) changes the action to a token and the reward source to a response-level judgement.

## What connects this week to earlier work?

You need the Q-learning target from week 4, the score-function idea from week 7,
and the chain rule from multivariable calculus.
Exploration changes the reward signal used to collect experience.
Meta-learning changes the initial parameters or internal state used to learn from experience.
A better exploration curve is not automatically evidence of better adaptation.
For that, hold the task distribution and adaptation data budget fixed.

## Analogy: a museum notebook and an adjustable bicycle

Imagine visiting a museum with a notebook of sketches.

| Everyday object | Mathematical counterpart |
|---|---|
| Exhibit and its permanent identification sketch | Observation `s` and frozen target `f(s)` |
| Your attempt to copy its sketch from memory | Predictor `g_phi(s)` |
| Squared copying error | RND intrinsic reward |
| Revisiting and practising the sketch | Predictor training on visited observations |

The analogy stops at usefulness: an unfamiliar poster is not necessarily an important exhibit.
RND error can reflect generalisation, capacity, or optimisation problems as well as novelty.
Changing irrelevant pixels can keep presenting difficult inputs to a finite predictor.
The target is random at initialisation, but deterministic when the same input returns.
That removes stochastic target noise; it does not prove every novelty signal is useful.
This distinction is central to [the RND paper, section 2.2](https://arxiv.org/html/1810.12894v1).

MAML is more like choosing bicycle adjustments before renting it to many riders.
The shared starting setup is `theta`; a rider's adjustments are the inner update.
Comfort after adjustment is the outer objective.
A setup that is best before adjustment need not be best after adjustment.
Unlike the analogy, real MAML differentiates through the adjustment procedure.

## Derivation: the gradient through one adaptation step

Let task `i` have training loss `L_i^tr` and separate validation loss `L_i^val`.
Let `theta` be a column vector of shared parameters and `alpha > 0` the inner learning rate.
Assume both losses are differentiable and the training loss is twice differentiable.
Treat the sampled training and validation data as fixed in this elementary derivation.

1. Adapt on training data:

   $$\theta_i'=\theta-\alpha\nabla_\theta L_i^{tr}(\theta).$$

2. Evaluate adaptation on validation data:

   $$F_i(\theta)=L_i^{val}(\theta_i'(\theta)).$$

3. Differentiate the inner update:

   $$\frac{\partial\theta_i'}{\partial\theta}=I-\alpha H_i,
   \qquad H_i=\nabla^2_\theta L_i^{tr}(\theta).$$

4. Apply the chain rule:

   $$\nabla_\theta F_i=(I-\alpha H_i)^\top
   \nabla_{\theta_i'}L_i^{val}(\theta_i').$$

5. Average these gradients across tasks, then descend on the shared parameters.

For a smooth scalar loss, the Hessian is symmetric, so the transpose is often omitted.
First-order MAML replaces the inner Jacobian by `I`.
It does **not** replace it by zero: detaching every adapted parameter loses the outer gradient.
For sampled RL trajectories, additional sampling-distribution terms require care;
the exact-expected-reward bandit in this project deliberately avoids that complication.
[MAML's original paper](https://proceedings.mlr.press/v70/finn17a.html) separates task adaptation from the objective evaluated after adaptation.

## Worked example 1: a curiosity bonus decays on one fixed observation

**Predict.** If the target stays fixed and a small gradient step improves the predictor,
the next visit should receive a smaller bonus.

Take a two-dimensional target `f(s) = (1, -1)` and prediction `g(s) = (0, 0)`.
For this calculation the prediction itself is the parameter vector `p`.
Use squared error `L(p) = ||p-f(s)||²` and learning rate `eta = 0.25`.

1. Initial error is `(−1, 1)`; the bonus is `1² + 1² = 2`.
2. The gradient is `2(p-f) = (−2, 2)`.
3. Update: `p' = (0,0) − 0.25(−2,2) = (0.5,−0.5)`.
4. New error is `(−0.5,0.5)`; the bonus is `0.25+0.25 = 0.5`.
5. If extrinsic reward is zero and curiosity weight is `0.1`, the shaped reward
   changes from `0.2` to `0.05` on the next evaluation of this observation.

**Interpret.** The residual shrank by a factor of four.
In this simple parameterisation, `p'−f = (1−2 eta)(p−f)`.
Thus the error contracts only when `0 < eta < 1`.
At `eta = 1`, the prediction overshoots to `(2,−2)` and the error remains `2`.
At `eta = 1.1`, the error becomes `2(−1.2)² = 2.88`.
A frozen target alone does not guarantee monotone optimisation.

## Worked example 2: why the Hessian factor matters

**Predict.** If adaptation already moves a parameter toward the task optimum,
the outer gradient must account for how much of its initial error remains.

Use a scalar quadratic task loss `L(theta) = (theta−3)²/2` for both stages.
Set `theta = 1`, `alpha = 0.25`, and outer learning rate `beta = 0.1`.

1. Training gradient: `theta−3 = −2`.
2. Adapted parameter: `theta' = 1 − 0.25(−2) = 1.5`.
3. Validation loss: `(1.5−3)²/2 = 1.125`, down from `2`.
4. Hessian: `H = 1`; inner Jacobian: `1−alpha H = 0.75`.
5. Outer gradient: `0.75(1.5−3) = −1.125`.
6. Shared-parameter update: `theta_new = 1 − 0.1(−1.125) = 1.1125`.

First-order MAML uses gradient `−1.5`, giving `theta_new = 1.15`.
At `alpha = 1`, every initial parameter reaches `3` in one inner step:
validation loss and the exact outer gradient are both zero.
That is an instructive edge case: a zero meta-gradient need not mean broken autograd.

## Bridge to the actual project

- [project/rnd.py](project/rnd.py): `RND.intrinsic` sums squared error over embedding
  coordinates; `predictor_loss` averages over observations. Distinguish these axes.
- `train_rnd_on_gridworld` measures the bonus before fitting the observation,
  combines intrinsic and extrinsic rewards, and masks terminal bootstrapping.
- [project/maml.py](project/maml.py): `_adapt` uses `create_graph=True` and the exact
  mean reward vector `mu`; it does not estimate that vector from pulled arms.
- `train_maml` averages post-adaptation losses across sampled tasks.
- [project/solutions.py](project/solutions.py): exercises 2 and 6 isolate predictor
  fitting and adaptation. Exercise 7 combines them with a task distribution.

Keep the grid comparison's seed, action-selection rule, and episode budget matched.
`make_hard_grid` creates an open 7×7 grid with `pits=()` and a distant positive-reward goal.
Measure extrinsic goal hits separately from total curiosity-augmented reward.
The MAML toy is an algebra demonstration with known task means, not a sample-efficiency benchmark.

## Retrieve and transfer

<details>
<summary>1. Why is a target network random only once?</summary>

Re-randomising it each visit makes the label change for the same input.
The predictor then chases a moving target, so residual error need not represent novelty.

</details>

<details>
<summary>2. If the density model changes rho(s) from 0.1 to 0.2, what is the pseudo-count?</summary>

Using `N_tilde = rho(1−rho_plus)/(rho_plus−rho)`, obtain
`0.1×0.8/0.1 = 0.8`. It is a model-derived count, so it need not be an integer.
See [Bellemare et al.](https://arxiv.org/abs/1606.01868) for the density-recoding construction.

</details>

<details>
<summary>3. What changes at test time in MAML versus an in-context learner?</summary>

MAML updates parameters using adaptation data. An in-context learner changes its
history/context or recurrent state while keeping learned weights fixed.
Both still need informative task-specific experience; “no gradients” does not mean “no learning input”.

</details>
