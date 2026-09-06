# Week 9 teaching guide — what PPO actually clips

Read this before workbook exercises 1–3; return to the code bridge before training.
The goal is to predict the direction of an update before trusting its loss curve.

## Connect to what you know

From weeks 7–8, recall the score-function gradient and an advantage baseline.
You should be able to explain why positive advantage increases a sampled action's probability.
PPO keeps that direction but limits the *incentive* for improving each sampled action too far.
It normally reuses a fresh rollout for several epochs, then collects another rollout.

## Analogy: a capped sales commission

Imagine a salesperson receives commission for increasing sales of a useful product.
Once the agreed increase is reached, extra sales earn no extra commission.
A harmful product has the reverse rule: enough reduction earns all available credit.

| Analogy | PPO quantity |
|---|---|
| Original sales level | Old action probability |
| New/original sales | Ratio $\rho$ |
| Product's usefulness | Signed advantage $\hat A$ |
| Commission cap | Clipped surrogate plateau |
| Shared sales effort | Shared neural-network parameters |

**Limit:** a commission cap cannot physically stop sales increasing.
Likewise, PPO does not project the policy into a probability box.
Other samples, shared parameters, or a large optimizer step can move an already-clipped ratio.
This is why a small clipping parameter is not a proof of small KL divergence.

## Derivation: keep the sign visible

Fix a rollout sample $(s,a)$, its old log probability, and its estimated advantage.
Treat the old policy and advantage as constants during optimization.
Assume the old policy assigns positive probability to the sampled action.

1. Define $\rho=\pi_\theta(a|s)/\pi_{old}(a|s)=\exp(\ell_\theta-\ell_{old})$.
2. The local surrogate is $\rho\hat A$.
3. At the old parameters, $\rho=1$ and

   $$\nabla_\theta(\rho\hat A)=\hat A\nabla_\theta\log\pi_\theta(a|s).$$

   This matches the sampled policy-gradient direction at that point.
4. Construct a pessimistic surrogate

   $$f(\rho,\hat A)=\min\{\rho\hat A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)\hat A\}.$$

5. Multiplication by a negative number reverses inequalities. Consequently,

   $$f(\rho,\hat A)=
   \begin{cases}
   \hat A\min(\rho,1+\epsilon),&\hat A\ge0,\\
   \hat A\max(\rho,1-\epsilon),&\hat A<0.
   \end{cases}$$

6. The implementation minimizes $L_\pi=-\operatorname{mean}(f)$.
   In an active branch, $\partial L_\pi/\partial\ell_\theta=-\rho\hat A$ for one sample.
   On a plateau that sample contributes zero policy-loss gradient.

This derives the two branches, not a TRPO-style monotonic-improvement theorem.
The action ratio changes measure at a fixed state; it does not replace the old state distribution.
See the clipped objective and its motivation in the [original PPO paper](https://arxiv.org/abs/1707.06347).

## Worked example 1 — the same ratio, opposite advantages

**Predict:** with $\epsilon=0.2$ and $\rho=1.5$, which advantage sign is clipped?
Use old probability $0.2$ and new probability $0.3$.

**Calculate:** $\rho=0.3/0.2=1.5$ and the clipped ratio is $1.2$.

| Advantage | Raw surrogate | Clipped candidate | Minimum | Loss |
|---:|---:|---:|---:|---:|
| $+2$ | $3$ | $2.4$ | $2.4$ | $-2.4$ |
| $-2$ | $-3$ | $-2.4$ | $-3$ | $3$ |

For $\hat A=+2$, increasing the log probability further gives no direct benefit.
For $\hat A=-2$, $\partial L/\partial\ell=-(1.5)(-2)=+3$.
Gradient descent lowers the log probability, correcting the excessive increase.

**Interpret:** clipping removes further improvement incentives, but keeps penalties for harmful moves.
It does not mean “every sample outside the interval has zero gradient.”

## Worked example 2 — the lower threshold

**Predict:** an action was bad; has reducing its probability enough made the loss flat?
Keep $\hat A=-2$, set old probability $0.2$, new probability $0.1$.

**Calculate:** $\rho=0.5$, clipped ratio $0.8$.
The two candidates are $-1$ and $-1.6$; their minimum is $-1.6$.
The loss is $1.6$ and its local gradient is zero.
Now change only the advantage to $+2$: the candidates become $1$ and $1.6$.
The loss is $-1$ and $\partial L/\partial\ell=-1$.
Gradient descent increases the probability of this good action.

**Interpret:** the negative-advantage curve is flat *below* $1-\epsilon$.
A zero advantage makes the policy-loss contribution zero for every ratio.
At the active kink, an autodiff library chooses a valid branch/subgradient convention.

## Worked example 3 — value clipping is not always strictly worse

**Predict:** does leaving the value clipping interval always increase loss above MSE?
Let $V_{old}=0$, $V=1$, and the value clip radius be $0.2$.

**Calculate:** the clipped prediction is $0.2$.
For target $R=2$, raw squared error is $1$ and clipped squared error is $3.24$.
The helper returns $\tfrac12\max(1,3.24)=1.62$, above half-MSE $0.5$.
For target $R=0$, errors are $1$ and $0.04$.
The helper returns $0.5$, exactly the raw half-MSE.

**Interpret:** clipping is pessimistic about improvement; an already-worse prediction remains penalized.
Policy probability ratios and value predictions have different units, even if a codebase reuses a radius.

## Code bridge

- [solutions.py](project/solutions.py): `ppo_clip_loss` returns the negative mean surrogate;
  feed the four cases above as a batch and predict the individual contributions first.
- `ppo_value_clip_loss` includes a factor of $1/2$; account for it when comparing to notes.
- `approx_kl` computes the nonnegative sample expression $\rho-1-\log\rho$.
  Its average is an estimator, not a hard constraint or a guarantee about every state.
- [agent.py](project/agent.py): `PPOAgent.update_minibatch` combines actor, critic, entropy,
  and gradient clipping; a flat actor term does not imply all parameters stop moving.
- [train.py](project/train.py): `_compute_gae` connects this week to week 8.
  Trace which tensors are stored from the rollout and reused without updating their reference.

Training improvement on a particular short run is an empirical result, not an algebraic invariant.
Separate “the loss matches these examples” from “three independent seeds improve.”

## Retrieval and transfer

<details>
<summary>1. If epsilon is 0.2, must every new/old probability ratio lie below 1.2?</summary>

No. The objective changes beyond the threshold; the policy is not projected or constrained there.
Other gradient terms and optimizer steps can move ratios further.
</details>

<details>
<summary>2. A bad action has ratio 1.4. Should the negative-advantage loss be flat?</summary>

No. Its probability increased, which is harmful. The raw branch remains active and gradient descent reduces its probability.
</details>

<details>
<summary>3. Why are old-policy KL and reference-policy KL different in RLHF?</summary>

The old rollout policy changes each collection cycle. The reference policy is usually a fixed anchor.
Small changes from many successive old policies can still produce a large total change from the reference.
</details>

Arithmetic checks: both signs at ratios 0.5 and 1.5, and both value-loss targets above,
were recomputed with Python during this teaching review.
