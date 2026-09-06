# Week 16 teaching guide — learn a simulator, then question its predictions

Use this guide before Gaussian KL, lambda-return, and the world-model exercises.

## Connect to what you know

Bellman targets bootstrap beyond observed rewards; a world model additionally predicts unobserved transitions.
An RSSM combines recurrent memory with a stochastic latent state.
Real observations update a posterior; imagined rollouts use only the learned prior.
The critic remains important because imagination stops at a finite horizon.

## Analogy: a flight simulator built from flight logs

The simulator's memory of recent controls maps to deterministic state $h_t$.
Uncertain hidden conditions map to stochastic latent $z_t$.
A new instrument reading maps to observation $o_t$, which updates the posterior.
Flying without new real readings maps to imagination under the prior.
The actor chooses controls; reward and continuation heads predict the consequences.

**Limit:** collecting more simulator hours does not repair a simulator with incorrect physics.
A controller can learn to exploit its errors, especially outside the recorded flight envelope.
Low observation reconstruction error alone does not establish accurate action consequences or long-run value.
Latent predictions from observational human data are not automatically causal intervention models.

## Derivation: where the world-model objective comes from

Write $y_t=(o_t,r_t,c_t)$ for consistently indexed prediction targets.
Condition on the action history, and let $q(z_{1:T}|y,a)$ be a posterior approximation.
Assume the model's conditional factorization and positive densities where the posterior has support.
Use $z$ here to avoid confusing the latent with an environment state named s.

1. Insert the posterior density inside the evidence integral:

   $$\log p_\theta(y|a)=\log\mathbb E_q\left[\frac{p_\theta(y,z|a)}{q_\phi(z|y,a)}\right].$$

2. Apply Jensen's inequality to the concave logarithm:

   $$\log p_\theta(y|a)\ge\mathbb E_q[\log p_\theta(y,z|a)-\log q_\phi(z|y,a)].$$

3. Factor the sequence using deterministic recurrent state $h_t$ and stochastic $z_t$.
4. Collect observation, reward, and continuation log likelihoods, and prior/posterior terms:

   $$ELBO=\sum_t\mathbb E_q[\log p(o_t|h_t,z_t)+\log p(r_t|h_t,z_t)+\log p(c_t|h_t,z_t)]
   -\sum_t\mathbb E_q[KL(q(z_t|h_t,o_t)\|p(z_t|h_t))].$$

5. Negate to obtain a loss: prediction losses plus KL.
   Gaussian likelihoods with fixed variance yield scaled squared errors;
   a Bernoulli continuation likelihood yields binary cross-entropy.
6. A KL coefficient $\beta\ne1$, free-nats clipping, or KL-balancing stop-gradients changes training.
   Those modifications should not be presented as the unchanged exact ELBO derivation.

This is an ELBO derivation under the stated factorization, not a guarantee of an accurate simulator.
See [Dreamer](https://arxiv.org/abs/1912.01603) for learning behavior in latent imagination.

## Worked example 1 — Gaussian disagreement and KL direction

**Predict:** if posterior and prior disagree in mean and variance, should swapping their roles preserve KL?
Use posterior $q=\mathcal N(1,1)$ and prior $p=\mathcal N(0,4)$, where the second number is variance.

**Calculate:** for scalar Gaussians,

$$KL(q\|p)=\log\frac{\sigma_p}{\sigma_q}
+\frac{\sigma_q^2+(\mu_q-\mu_p)^2}{2\sigma_p^2}-\frac12.$$

With $\sigma_q=1$, $\sigma_p=2$,
$KL(q\|p)=\log2+(1+1)/8-0.5=0.443147$ nats.
In the reverse direction,
$KL(p\|q)=\log(1/2)+(4+1)/2-0.5=1.306853$ nats.
For $q=p$, every dimension contributes $\log1+(1+0)/2-0.5=0$.

**Interpret:** KL is asymmetric; a broad prior and narrow posterior are not interchangeable.
Diagonal multidimensional KL sums coordinate contributions; a batch reduction is a separate operation.
For equal unit variances and means 1 versus 0, KL would be $1/2$, not zero.

## Derivation bridge — lambda returns in imagination

For horizon H, supply rewards $r_0,\ldots,r_{H-1}$, values $V_0,\ldots,V_H$,
and continuation probabilities $c_t\in[0,1]$.
Set $G_H=V_H$ and work backward:

$$G_t=r_t+\gamma c_t[(1-\lambda)V_{t+1}+\lambda G_{t+1}].$$

The bracket mixes one-step bootstrapping and a longer return.
At $\lambda=0$ it is the one-step target; at $\lambda=1$ it uses all imagined rewards plus the final bootstrap.
At a true terminal transition $c_t=0$, the target is exactly $r_t$.

## Worked example 2 — a terminal boundary stops propagation

**Predict:** can a large terminal-state value contaminate an earlier target?
Take $r=(1,2)$, $V=(4,3,10)$, $c=(1,0)$, $\gamma=0.9$, $\lambda=0.5$.

**Calculate:** initialize $G_2=10$.
Then $G_1=2+0.9(0)[0.5(10)+0.5(10)]=2$.
Next $G_0=1+0.9[0.5(3)+0.5(2)]=3.25$.
Advantages are $G-V=(-0.75,-1)$.
At $\lambda=0$, $G_0=1+0.9(3)=3.7$.
At $\lambda=1$, $G_0=1+0.9(2)=2.8$.

**Interpret:** the terminal mask removes $V_2$, but at lambda below 1, $V_1$ still enters the earlier target.
For the sampled first action with probability 0.6, the REINFORCE loss contribution is
$-\log(0.6)(-0.75)=-0.383119$.
This scalar is a loss value; the gradient is $-A\nabla\log\pi$, with advantage detached.
A negative advantage makes gradient descent reduce the sampled action's probability.

## Worked example 3 — small one-step model errors can affect many rewards

**Predict:** does a 0.1 state error per step necessarily produce total return error $0.1H$?
Consider true deterministic dynamics $x_{t+1}=x_t$, $x_0=0$.
The model predicts $\hat x_{t+1}=\hat x_t+0.1$.
Let the reward function be exactly known: $r_t=x_t$, with no discount for this finite example.

**Calculate:** true states and rewards are all zero.
Model states at times 0–3 are $(0,0.1,0.2,0.3)$.
For four rewards, the model return is $0+0.1+0.2+0.3=0.6$.
In general, its H-step error is
$0.1\sum_{t=0}^{H-1}t=0.1H(H-1)/2$.
For H=10 this is 4.5, although the local transition error is only 0.1.

**Interpret:** transition error changes future states, which can change many later rewards.
This toy has quadratic growth, not a universal model-error bound.
Actual bounds require reward scale, transition-error metric, distributional assumptions, and policy shift.
Short rollouts branched from real states are motivated by this tradeoff; see [MBPO](https://arxiv.org/abs/1906.08253).

## Code bridge and what the toy does not yet demonstrate

- [world_model.py](project/world_model.py): `RSSM.observe_step` uses a real observation;
  `imagine_step` uses the prior without that correction. `kl_gaussian` implements the direction in example 1.
- [train.py](project/train.py): `_world_model_update` returns reconstruction, reward, continuation,
  KL, and total losses; `_lambda_return` expects H rewards and H+1 values.
- `_imagine_from_states` samples discrete actions and records log probabilities for the actor loss.
  The training loss uses detached returns/advantages for the REINFORCE update.
- [solutions.py](project/solutions.py): exercise 5 demonstrates KL balancing separately from the training loop.
  Exercise 6 reports first/last evaluation return, not matched imagined/real trajectories.
  Exercise 7 currently returns final evaluation return, not the four world-model loss components.
- The collection policy reinitializes latent state for each observation; evaluation passes a zero
  previous-action vector. These simplify recurrent state handling and limit claims about a full Dreamer agent.

Do not use the labels of exercises 6–7 as evidence that their advertised diagnostics were measured.
A proper imagined/real comparison needs the same initial history, matched actions, and a common observable representation.

## Retrieval and transfer

<details>
<summary>1. Which distribution supplies the latent during imagination: posterior or prior?</summary>

The prior, because future real observations are unavailable. A posterior state inferred from real history can initialize the imagined rollout.
</details>

<details>
<summary>2. Does more imagined data always make the real policy better?</summary>

No. It can reduce optimization noise while amplifying a systematic model error. Evaluate on real or independently held-out environment rollouts and compare horizons.
</details>

<details>
<summary>3. With lambda=1 and horizon H, is the return necessarily a full Monte Carlo return?</summary>

No. It still bootstraps from V_H when the horizon ends before true termination. It is Monte Carlo to the end only when the rollout reaches a terminal boundary with no remaining bootstrap.
</details>

Arithmetic checks: both KL directions, lambda endpoints/terminal masks, actor loss, and model-error sums were recomputed in Python.
