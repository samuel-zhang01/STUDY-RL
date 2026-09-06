# Week 22 teaching guide: derive the constraint price and check its sign

Read with [the notes](notes.tex) and [the workbook](workbook.ipynb).
The examples are small constrained optimisation problems, not safety certificates for trained agents.

## Connection and prerequisites

Recall expected discounted return, policy gradients, and PPO from weeks 7–9.
Add a second signal: nonnegative cost `c`, with its own return `J_c` and budget `d`.
A reward is something to maximise; a cost limit is a constraint on what is acceptable.
Before using a budget, decide whether it measures discounted cost per episode,
undiscounted cost per episode, average cost per step, or a probability of failure.
These are different quantities even when all use the word “cost”.

## Analogy: a warehouse with an energy budget

A warehouse earns money from orders and consumes energy to fulfil them.
An internal energy price rises when the warehouse exceeds its allocation.

| Analogy | CMDP quantity |
|---|---|
| Income | Reward return `J_r(pi)` |
| Energy consumed | Cost return `J_c(pi)` |
| Energy allowance | Constraint budget `d` |
| Internal price per unit of energy | Nonnegative multiplier `lambda` |
| Profit after charging for energy | `J_r − lambda J_c` |

The price can guide behaviour toward the budget, but a noisy controller may overshoot.
A budget on expected energy does not guarantee every shift consumes less than the allowance.
If no feasible schedule exists, increasing the price indefinitely does not create one.
The analogy also explains complementary slackness: a nonbinding resource can have zero price.

## Derivation: ascent for the policy, descent for the multiplier

Consider `max_pi J_r(pi)` subject to `J_c(pi)<=d`.
Use the reward-maximisation convention throughout:

$$L(\pi,\lambda)=J_r(\pi)-\lambda[J_c(\pi)-d],\qquad\lambda\ge0.$$

1. For a violating policy, `J_c−d>0`, minimising over `lambda` sends `L` downward.
   Thus `max_pi min_lambda L` excludes violating policies in the ideal problem.
2. At fixed multiplier, the policy ascends:

   $$\theta' = \theta + \eta_\theta(\nabla J_r-\lambda\nabla J_c).$$

3. For a policy-gradient implementation, the combined advantage is
   `A_total=A_r−lambda A_c` before any additional normalisation.
4. The multiplier derivative is `partial L/partial lambda=−(J_c−d)`.
   Gradient **descent** therefore gives

   $$\lambda'=[\lambda-\eta_\lambda(-(\hat J_c-d))]_+
   =[\lambda+\eta_\lambda(\hat J_c-d)]_+.$$

5. At an appropriate optimum, feasibility and complementary slackness require
   `J_c<=d`, `lambda>=0`, and `lambda(J_c−d)=0`.

An increasing multiplier under a violation is still a descent step on this `L`.
Do not call it ascent on the displayed Lagrangian.
Finite discounted CMDPs have a linear occupancy formulation; neural parameterisations
and approximate alternating updates do not inherit an unconditional convergence guarantee.
[CPO](https://proceedings.mlr.press/v70/achiam17a.html) studies constrained policy updates and their approximation bounds.

## Worked example 1: the optimal constrained policy may mix actions

One state and one decision. The safe action has reward `2`, cost `0`.
The risky action has reward `6`, cost `2`. Budget is `d=0.5`.
Let `p` be the probability of selecting the risky action.

**Predict.** Always taking the risky action is infeasible, but occasionally taking it
may improve expected reward while respecting the expectation constraint.

1. Reward: `J_r(p)=2(1−p)+6p=2+4p`.
2. Cost: `J_c(p)=0(1−p)+2p=2p`.
3. Feasibility gives `2p<=0.5`, so `p<=0.25`.
4. Reward increases with `p`; the optimum is `p*=0.25`.
5. Optimal return is `2+4×0.25=3`; cost is `2×0.25=0.5`.
6. Lagrangian is `2+4p−lambda(2p−0.5)`.
   Its slope in `p` is `4−2 lambda`; indifference occurs at `lambda*=2`.
7. Complementary slackness holds: `2(0.5−0.5)=0`.

**Interpret.** Randomisation increases reward from the safe deterministic policy's `2` to `3`.
But 25% of individual decisions still incur cost `2`, above the numerical budget `0.5`.
An expected-cost constraint is not a per-decision bound.
If `d=0`, only `p=0` is feasible. If `d>=2`, the risky deterministic optimum is feasible.
Tight constraints do not universally require randomisation; this example does.

## Worked example 2: follow one multiplier and advantage calculation

**Predict.** A measured violation should raise the price of cost and make costly
advantages less attractive on the next policy update.

Take `lambda=0.5`, multiplier rate `eta=0.2`, observed cost `J_c=1.5`, and budget `d=1`.

1. Violation: `1.5−1=0.5`.
2. New multiplier: `lambda'=max(0,0.5+0.2×0.5)=0.6`.
3. For reward advantage `A_r=0.8` and cost advantage `A_c=1.5`,
   the unnormalised combined advantage changes from
   `0.8−0.5×1.5=0.05` to `0.8−0.6×1.5=−0.1`.
4. At ratio `1`, that action changes from weakly encouraged to discouraged.
5. If the next observed cost is `0`, the multiplier becomes
   `max(0,0.6+0.2(0−1))=0.4`.
6. From `lambda=0.05` with the same underspend it projects to `0`, never a negative price.

**Interpret.** The multiplier reacts to sampled cost and can oscillate.
A penalty coefficient that is updated only on positive violations cannot fall after underspending;
that is a different feedback law, as illustrated in the week 25 toy.

## A one-dimensional CPO feasibility check

Let proposed parameter change be `v`, cost constraint `v+0.2<=0`,
and trust region `v²/2<=0.02`.
The trust region is `−0.2<=v<=0.2`; only `v=−0.2` is feasible.
If current violation becomes `0.3`, the same region contains no feasible point.
This is why a recovery step differs from solving a feasible reward-maximisation problem.
For positive-definite `H`, `s=b^T H^(-1)b>0`, feasibility is `c<=sqrt(2 delta s)`.
The minimised QP dual has terms `+lambda delta−nu c`; reversing these signs changes the problem.

## Bridge to the actual project

- [project/agent.py](project/agent.py): `compute_gae` is called for reward and cost
  separately. `PPOLagrangian.update` combines the advantages with a minus sign.
- The implementation standardises reward advantages and divides the combination by
  `1+lambda`; its multiplier scale therefore differs from the raw derivation.
- [project/train.py](project/train.py): `_collect` accumulates episode costs for
  the multiplier; compare that aggregation with the discounted cost critic.
- [project/solutions.py](project/solutions.py): exercise 1 isolates the projected update;
  exercise 4 uses `lam` for a KL temperature and `nu` for the cost multiplier.

Name those quantities by role rather than assuming `lambda` always means the same thing.
The exercise 5 “CVaR” helper averages `ceil(alpha N)` sorted returns; when `alpha N`
is not an integer it approximates, rather than fractionally weights, the empirical tail.
For costs use the **upper** tail; for undesirable low returns use the lower tail.

## Retrieve and transfer

<details>
<summary>1. In example 1, what is the marginal reward value of a slightly larger budget?</summary>

For `0<d<2`, `p=d/2`, so optimal reward is `2+2d` and its derivative in `d` is `2`.
That matches the active multiplier's units: reward per extra unit of cost budget.

</details>

<details>
<summary>2. Can a large multiplier certify that the final policy is feasible?</summary>

No. It indicates costly behaviour was penalised. Measure the policy's cost independently
with uncertainty and the correct cost definition; infeasibility or failed optimisation may persist.

</details>

<details>
<summary>3. Must a cost advantage be nonnegative because costs are nonnegative?</summary>

No. `A_c=Q_c−V_c` is relative cost. A less costly-than-average action can have negative
cost advantage; subtracting `lambda A_c` then encourages it.

</details>
