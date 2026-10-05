# Week 02 teaching guide — turning the future into an equation

Use with [the workbook](workbook.ipynb) and [the notes](notes.tex).
The objective is to explain every term in a Bellman backup before running it.

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
- `exercise_7_gpi` stops when its policy is stable. With truncated evaluation,
  that alone is not a certificate of value accuracy: also inspect $\|T_*V-V\|$.
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
