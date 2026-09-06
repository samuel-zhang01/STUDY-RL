# Week 10 teaching guide — SAC from density to update direction

Use this guide for exercises 1–5 before running a continuous-control experiment.

## Connect to what you know

Recall Bellman bootstrapping, a target network, and the chain rule.
PPO uses a sampled log-probability gradient; SAC can differentiate through a sampled continuous action.
The new ingredient is an entropy term, whose value depends on using the correct action density.

## Analogy: choosing several commuting routes

A commuter prefers short journeys but keeps some variation among good routes.
The route cost maps to negative $Q$; willingness to vary maps to temperature $\alpha$.
Two skeptical journey-time forecasts map to the twin critics.
A reusable travel diary maps to the replay buffer.

**Limit:** entropy rewards distributional spread, not knowledge acquisition or fairness.
Randomizing equally between two dangerous routes is still dangerous.
Continuous entropy also changes with the units of the action coordinates.
A density can exceed one even though the probability of any measurable event is at most one.

## Derivation: why a softmax appears

First use finitely many actions so every operation is visible.
Fix a state, values $Q_a$, and $\alpha>0$.
Let $p_a\ge0$, $\sum_a p_a=1$, and use natural logarithms.

1. Maximize expected value plus entropy:

   $$F(p)=\sum_a p_aQ_a-\alpha\sum_a p_a\log p_a.$$

2. Add the normalization constraint with multiplier $\nu$:

   $$\mathcal L=F(p)-\nu(\sum_a p_a-1).$$

3. Differentiate: $Q_a-\alpha(\log p_a+1)-\nu=0$.
4. Rearrange: $p_a=\exp(Q_a/\alpha)\exp(-1-\nu/\alpha)$.
5. Normalize using $Z=\sum_a\exp(Q_a/\alpha)$:

   $$p_a^*=\frac{\exp(Q_a/\alpha)}{Z},\qquad V^*=\alpha\log Z.$$

6. The negative entropy is strictly convex, so the objective is strictly concave;
   the stationary point is the unique maximizer for finite values.

For continuous actions replace sums with integrals and require a finite partition function.
The practical Gaussian actor only approximates the unrestricted optimal distribution.
The soft Bellman target uses a sampled next action:

$$y=r+\gamma(1-d)\left[\min(Q'_1,Q'_2)-\alpha\log\pi(a'|s')\right].$$

Here $d=1$ denotes a true terminal transition; a time-limit truncation needs separate treatment.
See the [SAC paper](https://arxiv.org/abs/1801.01290) and [temperature-tuning formulation](https://arxiv.org/abs/1812.05905).

## Worked example 1 — temperature is a scale

**Predict:** with values $Q=(0,\log3)$ and $\alpha=1$, should the better action be deterministic?

**Calculate:** exponentiated values are $(1,3)$, so $p=(0.25,0.75)$.
Expected value is $0.75\log3=0.823959$.
Entropy is $-0.25\log0.25-0.75\log0.75=0.562335$.
Their sum is $1.386294=\log4=V^*$.

If $\alpha=0.5$, exponentiated values are $(1,9)$ and $p=(0.1,0.9)$.
If both $Q$ values and $\alpha$ double, the probabilities do not change.

**Interpret:** temperature is relative to reward/value scale.
In a finite action set, $\alpha\to0$ concentrates on maximum-value actions;
ties can remain shared, so “zero temperature selects a unique action” needs a unique maximum.

## Derivation bridge — squashing changes density

Let $u\sim\mathcal N(\mu,\sigma^2)$ and $a=\tanh u$.
Conservation of probability gives $p_A(a)\,da=p_U(u)\,du$.
Since $da/du=1-a^2$,

$$\log p_A(a)=\log p_U(u)-\log(1-a^2).$$

For independent action coordinates, sum the log-density and log-Jacobian terms.
If the environment action is $b+ca$ with $c>0$, subtract an additional $\log c$ per dimension.
Normalize the **density**: $\int p_A(a)\,da=1$.
There is no identity saying $\int\log p_A(a)\,da=0$.

## Worked example 2 — density at a concrete action

**Predict:** squashing compresses a neighborhood near the boundary; does its density rise or fall?
Take $\mu=0$, $\sigma=1$, and $u=\operatorname{atanh}(0.5)=0.549306$.

**Calculate:** $a=0.5$ and the Jacobian is $1-a^2=0.75$.
The Gaussian log density is
$-\tfrac12u^2-\tfrac12\log(2\pi)=-1.069807$.
The corrected log density is $-1.069807-\log0.75=-0.782125$.
Thus the action density is $0.457433$, versus Gaussian density $0.343075$.
Scaling the action to $2a$ changes the log density to $-1.475272$.

**Interpret:** the correction divides density by the local compression factor.
Use a stable log-Jacobian near $|a|=1$; directly computing $1-a^2$ may round to zero.

## Worked example 3 — terminal targets and actor-gradient sign

**Predict:** can entropy make a terminal transition bootstrap?
Take $r=1$, $\gamma=0.9$, next critics $(4,6)$, $\log\pi=-2$, and $\alpha=0.2$.

**Calculate:** the soft next value is $4-0.2(-2)=4.4$.
For $d=0$, $y=1+0.9(4.4)=4.96$.
For $d=1$, $y=1$, regardless of either critic or entropy.

Now take a deterministic toy actor $a=\tanh\mu$, $Q(a)=-a^2$, and no entropy.
The minimized actor loss is $L=-Q=\tanh^2\mu$.
At $\mu=\operatorname{atanh}(0.5)$,
$\partial L/\partial\mu=2(0.5)(1-0.25)=0.75>0$.
A step of size $0.1$ gives $\mu_{new}=0.474306$, closer to zero.

**Interpret:** a **positive** loss gradient at positive $\mu$ produces the desired decrease.
The sign of a loss gradient differs from the direction of the gradient-descent step.

## Code bridge

- [solutions.py](project/solutions.py): `tanh_gaussian_log_prob` sums over action dimensions;
  `soft_bellman_target` applies the terminal mask after constructing the soft value.
- `auto_temperature_loss` detaches `logp` because this step optimizes temperature.
  Its gradient in $\log\alpha$ is $\alpha(H-H_{target})$ using sampled $H=-\log\pi$.
  Entropy below target therefore increases $\alpha$ under gradient descent.
- [agent.py](project/agent.py): `TanhGaussianPolicy.sample` reparameterizes the action;
  `SACAgent.update` separates frozen critic targets from differentiable actor evaluations.
- [train.py](project/train.py): inspect action scaling before comparing reported entropy.

The twin minimum reduces one overestimation mechanism; it is not a calibrated lower confidence bound.
Purely offline SAC can exploit unsupported actions even with twin critics; week 14 addresses that problem.

## Retrieval and transfer

<details>
<summary>1. The action space doubles in width. Does differential entropy stay the same?</summary>

No. In one dimension, scaling by two adds log 2 to entropy and subtracts log 2 from log density.
The temperature target must be interpreted in the same action coordinates.
</details>

<details>
<summary>2. Why does a terminal SAC target ignore a huge next-state Q?</summary>

There is no future return after true termination; multiplying the full bootstrap term by 1-d makes the target equal the immediate reward.
</details>

<details>
<summary>3. Is off-policy replay itself evidence that the environment changed?</summary>

No. Replay intentionally contains data from older policies. Nonstationary transitions/rewards and inadequate coverage are separate problems requiring separate diagnostics.
</details>

Arithmetic checks: Gibbs probabilities, density transformation, both terminal masks,
and the analytic toy gradient were independently recomputed with Python.
