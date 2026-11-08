# Week 02 teaching guide — turning the future into an equation

**Course navigation:** [Week 01](../week-01-mdps-bandits/teaching_guide.md) ← **Week 02 / 26** → [Week 03](../week-03-monte-carlo/teaching_guide.md)

[Routes](../docs/LEARNING_MAP.md) · [All weeks](../docs/STUDY_GUIDE.md#week-by-week-teaching-guides) · [Notation](../docs/GLOSSARY.md) · [Progress](../docs/PROGRESS.md)

[Lecture PDF](notes.pdf) · [Workbook](workbook.ipynb) · [Extended reading PDF](extended_reading.pdf)

**You are here:** Why does a backup propagate information about the future?

**Recall first:** [Week 01](../week-01-mdps-bandits/teaching_guide.md). If unfamiliar, use the [repair tasks](../docs/LEARNING_MAP.md#if-a-prerequisite-is-missing).

**Visual anchor:** [See the connected diagram](../docs/CONCEPT_ATLAS.md#reward-return-and-value). Use it to explain the mechanism before the derivation.

Use with [the workbook](workbook.ipynb) and [the notes](notes.tex).
The objective is to explain every term in a Bellman backup before running it.

**Statistics companion:** [A Bellman expectation need never occur on a trajectory](../docs/statistics/01-probability-conditioning.md) · [Related foundation](../docs/statistics/02-expectation-variance.md) · [Textbook source map](../docs/statistics/SOURCES.md).

**Bounded first route:** Calculate one discounted return, evaluate one fixed policy, then perform one improvement step. Treat contraction bounds as a second pass after the table update is clear.

## Learning bridge

Follow [the staged learning path](../docs/LEARNING_PATH.md) when a topic is new.

- **Carry forward:** Retrieve the week 1 distinction between an immediate reward and an average outcome.
- **Add one difficulty:** Now a delivery action changes where you go. A known map lets you calculate the value of the next state before sampling trips.
- **Try after the worked example:** Open [the two-step delivery](../interactive/index.html#bellman) and change gamma from 0.9 to 0.4. Predict whether stop for 2 or continue for 4 one step later wins; then implement `discounted_return` in [the coding exercises](../learning_projects/exercises.py).

<details><summary>Check your prediction</summary>

Stop wins: continue is worth 0+0.4×4=1.6. The delivery reward stayed 4; the objective’s weight on delay changed.

</details>

**Carry it onward:** [Monte Carlo](../week-03-monte-carlo/teaching_guide.md) will keep the same return but replace known model averages with completed trips.


## Mechanism at a glance

```mermaid
flowchart LR
    V["Current values"] --> B["Average reward + discounted next value"]
    B --> N["New values under fixed policy"]
    N --> I["Optional: improve the policy"]
```

**Read the diagram:** A Bellman evaluation backup averages using a specified policy and transition model. Choosing a better action is a separate policy-improvement operation.

## Before you start

From Week 01, recall $G_t=R_{t+1}+\gamma G_{t+1}$.
You need conditional expectation, matrix multiplication, and a maximum.
This week assumes known transition and reward laws; Week 03 replaces the
model's expectations with complete sampled episodes.

## Analogy: a route planner with a complete map

| Route-planning idea | RL object |
|---|---|
| Current junction | State $s$ |
| Chosen outgoing road | Action $a$ |
| Travel cost on that road | Negative immediate reward |
| Likelihood of a diversion | Transition probability $P(s'\mid s,a)$ |
| Best estimated journey from the next junction | Continuation value $V(s')$ |

A backup combines the first road with the best remaining journey.
**Limit:** a Bellman expectation averages possible diversions before choosing
an action. It does not let the driver choose after seeing the random outcome.
Also, the route planner has a model; model-free learning need not have one.

## Derivation: why the Bellman equation is true

Assume a finite stationary MDP, bounded rewards, and $0\le\gamma<1$.
Fix policy $\pi$, and define $V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]$.
Let $p(s',r\mid s,a)$ be the joint next-state/reward probability.

1. Split the return after its first reward:

$$V^\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s].$$

2. Given the next state, the Markov property and fixed policy identify the
   expected remaining return as $V^\pi(s')$.
3. Expand the expectation over action, next state, and reward:

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma V^\pi(s')].$$

4. Define $r^\pi(s)=\mathbb E[R_{t+1}\mid s]$ and transition matrix $P^\pi$.
   Collect all states into a column vector:

$$V^\pi=r^\pi+\gamma P^\pi V^\pi,
\qquad (I-\gamma P^\pi)V^\pi=r^\pi.$$

Use a linear solver; writing an inverse explains the algebra but is not a
request to compute the inverse numerically.
At a true terminal state set remaining return to zero.
A continuing self-loop that pays reward is not a terminal state.

## Example 1: solve a two-state policy by hand

**Predict:** will repeated small rewards make state B worth more than one reward?

There are two continuing states and one action. A gives reward 1 then moves
to B. B gives reward 2 and stays in B. Set $\gamma=0.5$.

$$P=\begin{bmatrix}0&1\\0&1\end{bmatrix},
\quad r=\begin{bmatrix}1\\2\end{bmatrix}.$$

The equations are $V_A=1+0.5V_B$ and $V_B=2+0.5V_B$.
Thus $0.5V_B=2$, so $V_B=4$ and $V_A=3$.
Check the residual: $(3,4)-(1,2)-0.5(4,4)=(0,0)$.

From $V_0=(0,0)$, synchronous sweeps give

| Sweep | $V_A$ | $V_B$ | Maximum change |
|---|---:|---:|---:|
| 1 | 1 | 2 | 2 |
| 2 | 2 | 3 | 1 |
| 3 | 2.5 | 3.5 | 0.5 |

**Interpretation:** the geometric series of future rewards is being propagated
one backup at a time. The values are estimates; the reward law has not changed.
If B instead pays 2 once and terminates, $V_B=2$ and $V_A=2$.

## Derivation: why repeated backups converge

For any values $V,W$, transition probabilities are nonnegative and sum to one.
Therefore

$$|(T_\pi V)(s)-(T_\pi W)(s)|
\le\gamma\sum_{s'}P^\pi(s,s')|V(s')-W(s')|
\le\gamma\|V-W\|_\infty.$$

For optimality backups, use $|\max_a x_a-\max_a y_a|\le\max_a|x_a-y_a|$.
The same bound follows. This is a contraction when $\gamma<1$.
For a proposed value vector $V$ and residual $e=\|T_*V-V\|_\infty$,

$$\|V-V^*\|_\infty\le e+\gamma\|V-V^*\|_\infty
\quad\Rightarrow\quad \|V-V^*\|_\infty\le\frac{e}{1-\gamma}.$$

A tiny update needs interpretation relative to $1-\gamma$.
This argument certifies values, not the statistical accuracy of an estimated model.

## Example 2: a policy can change without improving

**Predict:** if the greedy action changes, must value strictly increase?

At a state S, actions left and right both give reward 5 then terminate.
Then $Q^\pi(S,\mathrm{left})=Q^\pi(S,\mathrm{right})=5$ for every policy.
Changing from left to right changes the policy but leaves $V(S)=5$.

For a different S, safe gives 2 then terminates; risky gives 8 with probability
$1/4$ and 0 with probability $3/4$, then terminates.
The risky expected value is $(1/4)8+(3/4)0=2$; both actions again tie.
Choosing after observing which outcome occurred would incorrectly give 8.

**Interpretation:** policy improvement guarantees nondecrease. Strict
improvement needs a strict advantage, not merely a different action label.
Risk-neutral expected return does not distinguish equal-mean reward distributions.

## Bridge to this week's code

- [solutions.py](project/solutions.py), `exercise_1_bellman_operator`, creates
  `V_new` from `V`: that is a synchronous backup with one fixed input vector.
- `exercise_6_policy_evaluation_closed_form` uses `np.linalg.solve` and masks
  terminal transitions when constructing the continuation matrix.
- [common/tabular.py](../common/tabular.py) provides VI and PI used by exercises 2–3.
  Different tie-breaking policies can have the same optimal value vector.
- `exercise_7_gpi` stops successfully only when its policy is stable **and** the
  optimal Bellman residual is below `tol`. If it reaches `max_outer`, the returned
  values are the current approximation, rather than a convergence certificate.
- Modified policy iteration must check the Bellman residual as well as policy
  stability: a stable action choice after one evaluation sweep can still have
  inaccurate values. In a one-state, reward-1 loop with gamma 0.5, the policy
  never changes, but values must approach $1/(1-0.5)=2$ rather than stop at 1.
- Car transfers respect both source inventory and destination capacity. At
  inventory $(2,2)$ with capacity 2, requesting a transfer of two moves zero
  cars and pays zero transfer cost. The Poisson tail horizon is at least the
  inventory cap, preserving capped transition probabilities.
- [carrental.py](project/carrental.py), `CarRentalMDP.actions_list`, maps stored
  action indices to signed numbers of cars moved. A heatmap of indices is misleading.

The connection between exact DP and approximate TD is discussed in
[Tsitsiklis and Van Roy's analysis](https://web.mit.edu/~jnt/www/Papers/J063-97-bvr-td.pdf):
approximation changes the fixed point and the relevant convergence conditions.
The finite calculations and inequalities above are derived directly here.

## Choose your explanation depth: How a local backup controls a global error

Open the first layer for the idea, the second to calculate, and the third to check the assumptions. The worked examples above and investigation below provide the longer route.

<details class="learning-layer">
<summary>1 · Intuition</summary>

A Bellman backup asks what one step plus a continuation is worth. If every continuation estimate is wrong by at most e, averaging cannot enlarge that error, and discounting shrinks its effect. Repeating this reasoning connects a local operation to convergence of the whole value table.

</details>

<details class="learning-layer">
<summary>2 · Work a numerical case</summary>

Use discount 0.9 and a current worst-state error of 2. One exact backup has error at most 1.8; two have error at most 1.62. The actual errors may be much smaller. A line below this envelope passes the check even if its logarithmic slope is not exactly log(0.9).

</details>

<details class="learning-layer">
<summary>3 · Derive it and state the limits</summary>

For an exact finite discounted model, $|(TV)(s)-(TW)(s)|\le\max_a\gamma\sum_{s\prime}P(s\prime\mid s,a)|V(s\prime)-W(s\prime)|\le\gamma\|V-W\|_\infty$. Apply this repeatedly with $W=V^*$ to obtain the error envelope. The additional residual certificate is $\|V-V^*\|_\infty\le\|TV-V\|_\infty/(1-\gamma)$, from the triangle inequality and contraction. Approximate regression or a wrong transition model introduces an additional error term.

</details>

**Try it in Jupyter:** jump to [`exercise_5_contraction_factor` in the workbook](workbook.ipynb#cell-22), then run the setup cells above it. Plot observed errors and the geometric upper envelope separately. A zero previous error makes an error ratio undefined; it does not indicate that contraction failed.

## Notebook investigation: watch information move through a value table

Open [the actual Jupyter workbook](workbook.ipynb) and find **Exercises 1, 2 and 6**.
The [GitHub week folder](https://github.com/samuel-zhang01/STUDY-RL/tree/main/week-02-bellman-dp)
contains this notebook, its `project/` imports and tests. Follow the
[download and Jupyter workflow](../docs/LEARNING_MAP.md#download-and-open-the-jupyter-materials)
so the shared `common/` package is available.

Use exercise 1 with a zero value vector. Mark which states can receive the goal reward in one
transition; only those immediate predecessors can acquire that goal information after one synchronous
backup. Repeat the operator yourself, saving each full value vector before replacing it.

Display successive grids with one fixed colour scale. An independently rescaled colourbar can make
an early small value look as strong as a later large value. Alongside the grids, plot the maximum
Bellman residual on a log scale; omit exact zeros from a logarithm. Compare with exercise 6's linear
solve under the **same fixed policy**. Value iteration optimises, so comparing it to an arbitrary
policy's solution answers a different question.

Change gamma from 0.5 to 0.95. For residual 0.01, the exact-model error bounds become 0.02 and 0.2.
Higher discount preserves more distant rewards and makes the same residual a weaker certificate.

<details>
<summary>Check your interpretation: If two policy arrows differ but their Q-values tie, must one of the grid plots be wrong?</summary>

No. Equal action values allow different optimal policies. Compare values, residuals and tie-breaking before interpreting arrows.

</details>

## Statistics bridge: A Bellman expectation need never occur on a trajectory

**Model and unit.** Keep the policy and next-state values fixed. A state moves to either of two successors with probabilities one-half; their continuation values are 0 and 4. Immediate reward is 1 and gamma is 0.9.

**Calculate before running.** The two one-step targets are $1+0.9(0)=1$ and $1+0.9(4)=4.6$. Their mean is $2.8$ and variance is $[(1-2.8)^2+(4.6-2.8)^2]/2=3.24$. The Bellman expectation is 2.8 even though neither sampled target equals 2.8.

**Interpret the result.** Dynamic programming averages a specified transition model. Sampling from that model adds target noise; estimating the model adds another uncertainty source. A small Bellman residual in a wrong model does not prove the value is correct in the real environment.

**Change one assumption.** Change gamma to 0.5 and recompute the mean and variance. What extra check is needed if the transition probabilities were fitted from data?

<details><summary>Check the changed assumption</summary>

The targets become 1 and 3, with mean 2 and variance 1. Fitted transitions need held-out predictive checks or uncertainty/sensitivity analysis; algebra alone does not verify their probabilities.

</details>

Work through the [foundation lesson](../docs/statistics/01-probability-conditioning.md) and the statistics investigation near the end of the [workbook](workbook.ipynb). Textbook grounding: Siegrist §4.7, PDF p.507; §16.3, PDF p.1322; [edition and page details](../docs/statistics/SOURCES.md). These are course-created numerical fixtures, not new benchmark measurements.

## Retrieve and transfer

<details><summary>1. A Bellman residual is 0.01 at γ = 0.99. What error is certified?</summary>

At most $0.01/(1-0.99)=1$ in sup norm under the exact discounted model.
A small residual can still allow a substantial value error when gamma is near one.
</details>

<details><summary>2. Why is a paying absorbing self-loop different from termination?</summary>

The self-loop continues to supply rewards. Termination supplies no future return;
only the reward on the transition into termination is retained.
</details>

<details><summary>3. If PI and VI choose different arrows, which one is wrong?</summary>

Possibly neither. Compare their values and action values; ties allow multiple
optimal policies. A policy map alone cannot establish disagreement in performance.
</details>

Arithmetic check: local Python solved the two-state system, reproduced all three
sweeps and zero residual, and checked the equal-mean action examples.
