# Week 26: causal discovery across biological environments

Start here, then use [the lecture](notes.tex), [the workbook](workbook.ipynb),
[the annotated reading](extended_reading.tex), and [the source dossier](../research/causal_biology.md).
This week asks: **what can varied, noisy patient observations tell us about an intervention,
and what remains unknown even after a prediction model improves?**

The lesson was prompted by Synlico's *Causal AI for Drug Discovery: Learning from Patient Data*
discussion. Its [official Tech Week listing](https://www.tech-week.com/calendar/sf/events/causal-ai-for-drug-discovery-learning-from-patient-data-sf-techweek-5ccd23ec-afbe-4206-99a0-3208dd443476)
lists a virtual October 5, 2026 discussion on learning causal structure across environments
and task-aware enhancement. This bundle teaches public methods; it does not reproduce
Synlico's proprietary platform or summarize an event recording.

## Where this fits

Prerequisites: expectations, conditional probability, linear regression, train/validation/test
splits, and the overlap reasoning from Week 11. A little familiarity with directed graphs
helps, but the lecture introduces them. Weeks 11 and 12 cover logged-policy and examination
corrections; Week 14 covers unsupported actions; Week 25 covers audits. Their support and
evaluation principles carry over. The new material is causal graph identification,
environment-dependent mechanisms, biological measurement, and choosing enhancement for a
specified task. This is an adjacent causal-learning week, not a new deep-RL algorithm.

## One week, five two-hour sessions

| Session | Read and do | Deliverable |
|---|---|---|
| 1: define the question | 30 min guide; 45 min lecture motivation/formulation; 45 min workbook 1–2 | A target population, treatment, outcome, time horizon, confounder set, and calculated adjusted effect |
| 2: learn what graphs cannot decide | 45 min equivalence/latent-confounding lecture; 30 min Li–Jaber–Bareinboim reading; 45 min workbook 3 | Two observationally indistinguishable models with different intervention effects |
| 3: reason across environments | 40 min invariance/algorithm distinctions; 30 min ICP reading; 50 min workbook 4 and 6 | A patient split and an assumption table naming which mechanisms may change |
| 4: enhancement and leakage | 35 min implementation; 25 min task-driven augmentation and imputation readings; 60 min workbook 5 and 7 plus CLI | Training-only imputation/augmentation and a baseline/ablation table |
| 5: assess a claim | 30 min safety/matching chapters; 45 min workbook 8 and seed/shift extension; 45 min memo and speaker questions | One page separating estimation error, graph ambiguity, shift uncertainty, and experimental validation |

Spend any extra weekend time on *What If*, chapters 1–3, 6–9, and 13, rather than treating
this one week as comprehensive training in biomedical causal inference.
[The authors supply the book and code](https://miguelhernan.org/whatifbook).

## Analogy: different kitchens, one recipe mechanism

Imagine several kitchens make a dish. Oven temperature affects doneness; a camera records
the plate. One kitchen changes temperature, another changes lighting. You want to learn what
happens when you set the oven, not merely predict what the photograph looks like.

| Kitchen object | Mathematical object |
|---|---|
| Temperature deliberately set | Intervention `do(A=a)` replacing the assignment rule for `A` |
| Ingredient quality that affects both temperature choice and doneness | Pre-treatment confounder `C` |
| Temperature-to-doneness recipe | Structural equation `Y=f_Y(A,C,U_Y)` |
| Kitchen with a different measurement/assignment mechanism | Environment `E=e` with distribution `P^e` |
| Lighting makes the photograph more predictive in one kitchen | Environment-dependent nuisance proxy `S` |
| Unrecorded humidity affecting several quantities | Latent common cause `U` |

The analogy breaks when a kitchen changes the recipe itself: the target mechanism may
cease to be invariant. Biological tissues can differ in regulation, cell composition, timing,
and measurement. An environment label is evidence about possible changes, not proof of a
randomized intervention. Removing all environment effects can erase the very variation
that supports discovery.

## Four questions with four different answers

Prediction asks for `E[Y | X=x]`. Causal effect estimation asks for `E[Y(1)-Y(0)]` in a specified
population. Causal discovery asks which directed structures fit the observations and
assumptions. Drug-target validation asks whether a controlled perturbation produces the
expected response in a relevant biological context. A low held-out prediction error cannot
answer the other three by itself.

A **DAG** is a directed graph without directed cycles. An arrow encodes direct dependence
in a proposed mechanism, not a correlation threshold. An **SCM** adds assignment equations
and a disturbance distribution. In a simple DAG `C → A → Y` with `C → Y`, the path
`A ← C → Y` confounds the observed association. Conditioning on appropriate pre-treatment
`C` blocks that backdoor path. A mediator on `A → M → Y` answers a different question if
adjusted for, and conditioning on a collider `A → K ← Y` can create dependence.

`do(A=a)` replaces the equation generating treatment, while `A=a` selects observations
generated by the existing equation. A molecular knockout and a recorded medication are
different interventions with different outcomes and populations; neither should silently
stand in for the other.

## Central derivation: standardization identifies an ATE under stated assumptions

Let a patient have binary treatment `A`, pre-treatment categorical confounder `C`, observed
outcome `Y`, and potential outcomes `Y(0),Y(1)`. Define the target as
`ATE = E[Y(1)] - E[Y(0)]` under the target population's distribution `P(C)`.
Use the same treatment version and follow-up definition for every patient.

Assume consistency (`Y=Y(a)` when `A=a`, with no relevant interference), conditional
exchangeability (`Y(a)` independent of `A` given `C`), and positivity
(`P(A=a | C=c)>0` for each `a,c` with target mass). Measurement must be adequate for these
assumptions; adding a noisy proxy for an omitted cause need not establish exchangeability.

For either treatment level `a`:

1. The law of total expectation gives
   `E[Y(a)] = sum_c E[Y(a) | C=c] P(C=c)`.
2. Exchangeability permits
   `E[Y(a) | C=c] = E[Y(a) | A=a,C=c]`.
3. Consistency permits
   `E[Y(a) | A=a,C=c] = E[Y | A=a,C=c]`.
4. Positivity ensures the conditional observed expectation is defined for every target stratum.
5. Subtract the expressions for `a=1` and `a=0`:

   $$\mathrm{ATE}=\sum_c\{\mathbb E[Y\mid A=1,C=c]-\mathbb E[Y\mid A=0,C=c]\}P(C=c).$$

Replacing stratum expectations and target proportions with sample means gives an estimator,
not another identification theorem. [Exercise 2](project/solutions.py) implements empirical
standardization and raises an error for an unobserved treatment cell. The target distribution
here is the sample's confounder distribution; a new target population needs its own weights
and transport assumptions. This derivation follows the standard identification conditions
introduced in [Hernán and Robins](https://miguelhernan.org/whatifbook).

## Numerical example 1: confounding reverses a decision

**Teaching fixture, not a patient study.** There are 100 patients in each severity stratum.
The outcome is a benefit score; larger is better. Each treated patient's improvement is one
unit. Severe patients happen to have a lower starting score and receive treatment more often.

| Stratum | Treated count/mean | Untreated count/mean | Within-stratum effect |
|---|---|---|---|
| Mild `C=0` | 20 / 3 | 80 / 2 | `3-2=1` |
| Severe `C=1` | 80 / 1 | 20 / 0 | `1-0=1` |

**Predict.** Will a pooled treated–untreated difference show the one-unit benefit?

**Calculate.** Treated mean is `(20×3+80×1)/100 = 140/100 = 1.4`.
Untreated mean is `(80×2+20×0)/100 = 160/100 = 1.6`.
The naive difference is `1.4-1.6=-0.2`. Standardization uses the same target severity weights
for both treatment arms: `0.5(3-2)+0.5(1-0)=1`.

**Interpret.** Treated and untreated patients have different severity mixtures. The arithmetic
exposes that association problem; causal interpretation still needs the assumptions above.
If all severe patients are treated, `E[Y | A=0,C=1]` is unobserved. The estimator must stop;
imputation cannot certify the missing counterfactual.

## Numerical example 2: the same observations support two directions

**Teaching fixture.** Let `U,V` be independent standard normal disturbances and `rho=0.8`.
Model F is `X=U`, `Y=0.8X+0.6V`. Model R is `Y=U`, `X=0.8Y+0.6V`.

**Predict.** Can an unlimited observational sample distinguish these models?

**Calculate.** In F, `Var(X)=1`, `Cov(X,Y)=0.8`, and
`Var(Y)=0.8²×1+0.6²×1=0.64+0.36=1`.
R has the same covariance matrix `[[1,0.8],[0.8,1]]` and the same zero means.
Joint Gaussian distributions are determined by those means and covariances, so the
observational distributions coincide. Under `do(X=1)`, F has `E[Y]=0.8`; R keeps `Y=U`,
so `E[Y]=0`. The difference is `0.8` despite identical observational data.

**Interpret.** [Exercise 3](project/solutions.py) demonstrates identification limits, not a
failed optimizer. A latent common cause is another possible explanation. Directions require
additional assumptions, background knowledge, or informative interventions/environments.
If two environments change only a camera calibration, they may add no identifying signal
about the direction between the biological variables.

## Numerical example 3: a prettier reconstruction can change the wrong dependency

**Teaching fixture.** Independent mean-zero true markers `G1,G2` each have variance one.
Suppose a smoother outputs `H1=(G1+G2)/2` and `H2=(G1+G2)/2`.

**Predict.** Does reduced variance preserve the absence of a gene–gene relationship?

**Calculate.** Both output variances are `(1+1)/4=0.5` and their covariance is `0.5`.
Their correlation is `0.5/sqrt(0.5×0.5)=1`, while the original covariance was zero.

**Interpret.** Sharing values can manufacture dependencies. This exact fixture is our
counterexample, not a reproduction of any single-cell method.
[Andrews and Hemberg](https://f1000research.com/articles/7-1740) evaluated false signals
after imputation, motivating negative controls alongside reconstruction or prediction scores.

## Discovery algorithms: know what the output means

| Method | Question and assumptions | Typical output | This project |
|---|---|---|---|
| PC | Conditional independences, acyclic causally sufficient graph, Markov/faithfulness assumptions, suitable tests | CPDAG, an equivalence-class summary | Not implemented |
| FCI | Conditional independences with latent confounding; graphical/statistical assumptions remain | PAG with potentially unresolved endpoints | Not implemented |
| S-FCI | Cross-domain observational/interventional constraints using selection diagrams and S-Markov theory | Cross-domain equivalence-class constraints | Not implemented |
| ICP | Which predictor sets leave the target conditional mechanism invariant under its assumptions? | Intersection of accepted sets; formal error control needs valid tests | Not implemented |
| Our residual summary | Do fitted residual means/variances differ across training environments? | Two descriptive spreads | Implemented, with no p-values or error control |

PC and FCI are background in [Joint Causal Inference](https://jmlr.org/papers/v21/17-123.html).
S-FCI is from [Li, Jaber, and Bareinboim](https://proceedings.neurips.cc/paper_files/paper/2023/hash/368cba57d00902c752eaa9e4770bbbbe-Abstract-Conference.html).
The target-parent view is developed by [Peters, Bühlmann, and Meinshausen](https://arxiv.org/abs/1501.01332).
Their results depend on formal assumptions, not merely having several batches of data.

Failure to reject residual change is weaker than showing invariance, and invariance alone may
leave multiple sets accepted. If the target equation changes, even its true parents need not
give stable predictions. Finite-sample tests, multiple comparisons, selection, measurement,
and hidden confounding each create separate uncertainty.

## Code bridge: one bounded experiment

Run from this week directory:

```bash
python -m project.train --seed 0 --output runs/result.json
python -m project.make_figures
```

[The generator](project/env.py) creates three environments with 240 patients each and three
technical measurement rows per patient. True markers `Z,C` generate `Y=1.5Z+0.4C+U_Y`.
A noncausal proxy `S=a_eY+U_S` changes coefficient across environments: `(2,0,-2)`.
Observed markers have additive measurement noise and 30% explicitly marked MCAR missingness.
These continuous markers are deliberately simpler than real count-based single-cell RNA data.

[Splitting](project/solutions.py) assigns every row for a patient to one partition, seals
environment 2, and divides patients in environments 0 and 1 into train and validation.
[Preprocessing](project/agent.py) fits imputation means and scales on training rows only.
There are no biological-zero-to-missing conversions. With varying numbers of rows per
patient, you should also decide how to weight the fitting procedure, not just evaluation.

[Training](project/train.py) compares raw ridge with a predefined family of nuisance
randomizations. Only proxy column 2 is randomized; targets are kept. It picks probability
`p∈{0,0.5,1}` using worst-environment **validation** MSE. This is a task-aware heuristic
whose validity relies on our known synthetic nuisance. It is neither a learned biological
generator nor Synlico's enhancement method, and is unsuitable as an unexamined preprocessing
step before causal discovery. Imputed/augmented values are model assumptions, not new patients.

For seed 0, the project produces:

| Quantity | Result |
|---|---:|
| Separate confounded fixture: true ATE | 1.0000 |
| Naive observed difference | 2.2106 |
| Adjusted estimate | 1.0009 |
| Raw ridge, environment 2 MSE | 5.2622 |
| Validation-selected randomization `p=0.5`, environment 2 MSE | 3.0479 |
| Ablation: omit proxy entirely, environment 2 MSE | 1.7676 |
| Training-target-mean baseline, environment 2 MSE | 3.6047 |

ATE and prediction are separate experiments. The model is never evaluated as a medication
policy. The ablation beats the selected enhancement in this unseen environment: validation
over source environments does not guarantee optimality for a new shift. The learning curves
optimize different training objectives, so compare methods on the same held-out metric.
The patient bootstrap interval describes sampling variability conditional on the fitted model;
it excludes training, graph-selection, and transport uncertainty.

## Questions to take to the scientific discussion

1. What exactly counts as an environment: patient, tissue, cell state, study, sequencing
   chemistry, or batch? Which biological and measurement mechanisms may differ?
2. Which conclusions are identified only up to an equivalence class? How are unresolved
   directions and latent confounding shown in the outputs?
3. What assumptions connect natural patient variation to the intervention being simulated?
   Does a gene intervention preserve those assumptions in the intended cell state?
4. What task trains the enhancement model: reconstruction, prediction, structure recovery,
   perturbation response, or a joint objective? What negative controls catch false edges?
5. Are enhancement fitting, hyperparameter selection, and evaluation split by patient and
   study? How do repeated cells affect uncertainty and effective sample size?
6. Which held-out perturbations or independent cohorts validate predicted response? Are
   failed predictions and context-specific limitations reported alongside successes?

These are questions to ask, not claims about the speakers' methods. Public material reviewed
for this lesson did not disclose enough detail to identify a Yang-specific enhancement
algorithm. His [UAI 2022 paper](https://proceedings.mlr.press/v180/yang22e.html) concerns noisy
sparse subspace clustering, a related noise/structure topic rather than proof of a proprietary
causal-enhancement system.

## Retrieve, calculate, and transfer

<details>
<summary>1. Retrieve: what changes under do(A=a)?</summary>

The assignment equation for `A` is replaced. Conditioning on `A=a` instead retains that
equation and selects observations; common causes can therefore have a different distribution.

</details>

<details>
<summary>2. Calculate: use 80% mild, 20% severe as the target population in example 1.</summary>

The standardized effect is `0.8(3-2)+0.2(1-0)=1`. Because the within-stratum effects are
equal here, changing target weights changes each potential-outcome mean but not their difference.
With effect modification, the target weights would change the ATE.

</details>

<details>
<summary>3. Transfer: test patients have appeared in the enhancement model's training cells. Is the split valid?</summary>

It is not an independent patient evaluation. All learned preprocessing, enhancement,
feature selection, and predictors must respect the patient split. Shared individuals,
specimens, or donors require an explicit grouping rule even if cell IDs differ.

</details>

<details>
<summary>4. Transfer: does a stable residual score identify a causal parent?</summary>

No. Our score has no formal test, may miss changes, and may accept several predictor sets.
The true target mechanism can also change. Formal ICP requires its own assumptions and
valid tests; PC/FCI/S-FCI answer different structure-learning questions.

</details>

<details>
<summary>5. Transfer: suppose target_shift=1 changes the test outcome mechanism.</summary>

The stable-feature model was trained for coefficient 1.5, while the test coefficient becomes
2.5. Omitting the proxy cannot correct that target-mechanism change. Run the CLI with
`--target-shift 1`; report all models rather than tuning the rule after seeing the test outcome.

</details>

All three numerical fixtures were independently checked in Python by
[the workbook](workbook.ipynb); fixture and leakage contracts are tested in [tests](tests/test_solutions.py).
