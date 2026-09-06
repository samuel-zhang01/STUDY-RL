# Week 08 teaching guide — blending horizons and controlling policy change

Use with [the workbook](workbook.ipynb) and [the notes](notes.tex).
Keep three ideas separate: the advantage estimate, the update direction, and its size.

## Before you start

Recall TD residuals from Week 04 and score-function actor updates from Week 07.
You need finite geometric sums, vector inner products, and basic matrix solves.
A2C uses a critic to estimate advantages. GAE controls how much future residuals
contribute. Natural gradients and TRPO control changes to action distributions.

## Analogy: checking a travel forecast at several checkpoints

| Forecast idea | Mathematical concept |
|---|---|
| Surprise after one road | TD residual $\delta_t$ |
| Combining surprises over several roads | n-step advantage |
| Giving distant checkpoints less influence | GAE factor $\gamma\lambda$ |
| Current forecast | Critic $V$ |
| Limiting how much route-choice frequencies change | Trust region in policy KL |

**Limit:** trust regions constrain probability distributions under specified state
weighting. They do not directly certify safe outcomes, and rare states can be
poorly controlled by a small average KL.

## Derivation: the telescoping identity behind GAE

Use rewards $r_t$ after action $a_t$, values $V_t=V(s_t)$, and discount $\gamma$.
For a continuing segment,

$$\delta_t=r_t+\gamma V_{t+1}-V_t.$$

Expand two successive terms explicitly:

$$\delta_t+\gamma\delta_{t+1}
=r_t+\gamma V_{t+1}-V_t
+\gamma r_{t+1}+\gamma^2V_{t+2}-\gamma V_{t+1}
=r_t+\gamma r_{t+1}+\gamma^2V_{t+2}-V_t.$$

The internal value terms cancel. Repeating this cancellation gives

$$\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}.$$

For $0\le\lambda<1$, average horizons with weights $(1-\lambda)\lambda^{n-1}$.
Assuming absolute convergence, the coefficient of residual $\delta_{t+k}$ is

$$(1-\lambda)\gamma^k\sum_{n=k+1}^{\infty}\lambda^{n-1}
=(1-\lambda)\gamma^k\frac{\lambda^k}{1-\lambda}
=(\gamma\lambda)^k.$$

Thus $\hat A_t=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}$, which gives the
backward recurrence $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$.
At $\lambda=1$, use the residual sum or a limit; direct substitution into
$(1-\lambda)\sum\lambda^{n-1}\hat A^{(n)}$ is not a valid $0\times\infty$ calculation.
See [Schulman et al., Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

## Example 1: compute every GAE entry

**Predict:** at lambda zero, can a later residual influence the first advantage?

Rewards $(1,2,3)$, values $(0.5,1,1.5)$, $\gamma=0.5$;
the third transition truly terminates, so next value is zero.
Residuals are

$$\delta_0=1+0.5(1)-0.5=1,
\quad\delta_1=2+0.5(1.5)-1=1.75,
\quad\delta_2=3-1.5=1.5.$$

At $\lambda=0.8$, $\gamma\lambda=0.4$:
$\hat A_2=1.5$;
$\hat A_1=1.75+0.4(1.5)=2.35$;
$\hat A_0=1+0.4(2.35)=1.94$.
Critic targets $\hat R=\hat A+V$ are $(2.44,3.35,3)$.

At $\lambda=0$, advantages equal $(1,1.75,1.5)$.
At $\lambda=1$, they are $(2.25,2.5,1.5)$;
adding values gives $(2.75,3.5,3)$, exactly the full discounted returns.

**Interpretation:** lambda changes the target for early states; the final
terminal target remains the observed final reward.
These critic targets are estimates, not additional observed rewards.

## Example 2: lambda one does not remove an unfinished rollout's bootstrap

**Predict:** does lambda one always mean a pure Monte Carlo return?

Now the same three transitions end only because collection stopped.
Set $V(s_3)=4$, with no terminal flag.
The last residual becomes $3+0.5(4)-1.5=3.5$.
At $\lambda=1$, advantages become $(2.75,3.5,3.5)$ and targets
$(3.25,4.5,5)$.

For the first target, direct expansion gives

$$1+0.5(2)+0.5^2(3)+0.5^3(4)=3.25.$$

The extra 0.5 compared with terminal MC is the final bootstrap contribution.
**Interpretation:** lambda one removes intermediate bootstrap terms by cancellation;
it does not eliminate the surviving end-of-segment value.

A true terminal zeros both the bootstrap and propagation across the boundary.
At a time-limit truncation in a continuing task, bootstrap from the final
observation but stop propagation into the reset episode. Those need two masks
or an equivalent carefully implemented correction.

## Derivation: the natural-gradient trust-region step

Let $g$ be the gradient of a local improvement surrogate, $u$ a parameter change,
and $F$ a positive-definite Fisher matrix, with KL budget $\delta>0$.
The local constrained problem is

$$\max_u g^Tu\quad\text{subject to}\quad\tfrac12u^TFu\le\delta.$$

The Lagrangian condition is $g-\eta Fu=0$, giving $u=F^{-1}g/\eta$.
Substituting into the active constraint gives

$$u=\sqrt{\frac{2\delta}{g^TF^{-1}g}}F^{-1}g.$$

This is a local quadratic solution, not an exact solution to nonlinear TRPO.
In large networks, solve $Fx=g$ through matrix-vector products rather than
explicitly inverting F; damping handles near-singular empirical Fishers.

## Example 3: a complete two-dimensional trust-region step

Take $F=\operatorname{diag}(4,1)$, $g=(2,1)$, and $\delta=0.01$.
Solve $Fx=g$: $x=(0.5,1)$; then $g^Tx=2$.
Scale $\sqrt{2(0.01)/2}=0.1$, so $u=(0.05,0.1)$.
Quadratic KL is $\tfrac12[4(0.05)^2+(0.1)^2]=0.01$.
Predicted surrogate gain is $2(0.05)+1(0.1)=0.2$.

**Interpretation:** the first direction is more sensitive under F and receives
a smaller parameter displacement. Actual policy KL still needs measurement.
TRPO's theoretical guarantee uses a maximum over states; its practical
average-KL constraint is an approximation. See
[Schulman et al., Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477).

For categorical distributions, Pinsker gives
$\mathrm{TV}(p,q)=\tfrac12\|p-q\|_1\le\sqrt{\mathrm{KL}(p\|q)/2}$.
At KL $\le0.001$, a single action/event probability changes by at most
$\sqrt{0.001/2}=0.022361$, or 2.24 percentage points; the L1 bound is 0.044721.
An average-KL bound yields an average event bound under the same fixed state law,
not a pointwise guarantee at every state or at a changed future state distribution.

## Bridge to this week's code

- [solutions.py](project/solutions.py), `compute_gae`, accepts one rollout and
  returns `(advantages, returns)`; reproduce Examples 1–2 with its `dones` array.
- `nstep_advantage` takes a separate final `bootstrap`; `conjugate_gradient`
  solves the matrix-vector system in Example 3.
- [train.py](project/train.py) collects `(time, environment, feature)` arrays
  and computes GAE separately for each environment before flattening.
- The training loop combines termination and truncation. It also clips sampled
  Gaussian actions and stores the clipped values; scoring those as ordinary
  Gaussian draws is a likelihood mismatch. Treat the trainer as a teaching
  prototype until boundary handling and action-distribution consistency are addressed.
- [agent.py](project/agent.py) trains A2C. The project supplies TRPO building
  blocks and notes, not a complete TRPO training implementation.

## Retrieve and transfer

<details><summary>1. Why is a value added back to the advantage for critic training?</summary>

The advantage subtracts the current state's value. Adding it back constructs
the corresponding estimated return target; the critic predicts returns, not advantages.
</details>

<details><summary>2. Does average KL below 0.001 bound every state's action change?</summary>

No. A rare state can have large KL while its contribution to the average is
small. The 2.24-point event bound applies pointwise only with a pointwise KL bound.
</details>

<details><summary>3. Does a successful backtracking step prove higher real return?</summary>

No. It checks empirical surrogate and empirical KL. Sampling error, critic error,
and the average-versus-maximum constraint gap remain.
</details>

Arithmetic check: local Python reproduced all GAE vectors, endpoint bootstraps,
the Fisher solve and quadratic budget, and both numerical Pinsker bounds.
