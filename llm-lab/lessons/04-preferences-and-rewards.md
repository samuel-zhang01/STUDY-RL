# Lesson 4 — From preference pairs to rewards and online policy updates

**Outcome:** derive DPO, interpret its numerical margin, and distinguish it from the
RLOO/GRPO rollout loop used for verifiable rewards.
Prerequisites: [lesson 1](01-llms-as-policies.md), [LoRA](02-real-lora-training.md),
[week 18: KL and RLHF](../../week-18-rlhf-ppo-llm/teaching_guide.md),
[week 19: DPO](../../week-19-dpo-zoo/teaching_guide.md),
[week 20: group RL](../../week-20-grpo-rloo-rlvr/teaching_guide.md), and
[week 21: reward hacking](../../week-21-reward-hacking-oversight/teaching_guide.md).

The new lab runs **offline LoRA DPO** on correct-versus-incorrect addition pairs.
The online RLOO/GRPO material below explains the next connection; its runnable implementation
is [week 20's separate toy trainer](../../week-20-grpo-rloo-rlvr/project/train.py).
The new pretrained-model lab does not claim to run a full GRPO training pipeline.

## Analogy: changing a menu without abandoning familiar choices

A restaurant wants customers to prefer its dishes while avoiding an abrupt change from its current menu.

| Menu analogy | Mathematical quantity |
|---|---|
| Current distribution over dishes | Trainable response policy `pi_theta` |
| Familiar starting menu | Frozen reference `q` |
| Value of serving a dish | Reward `r(x,y)` |
| Customer prefers one of two dishes | Chosen/rejected pair at the same prompt |
| Cost of changing the menu's distribution | `beta KL(pi||q)` |

The analogy explains the reward–drift trade-off. It does not make preferences objective truth.
Customers can disagree, and “correct versus wrong” is a deliberately narrow preference source.
DPO learns a **relative** change in likelihood; it need not increase the chosen answer's absolute probability.

## Derive DPO instead of memorising four log probabilities

Fix a prompt `x` and suppress it temporarily.
Assume `beta>0`, reference support wherever responses may receive policy probability,
and a finite partition function. First consider unrestricted response distributions,
not a finite-rank adapter with limited expressivity.
The regularised RL objective is

$$\max_\pi\left[\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log\frac{\pi(y)}{q(y)}\right],
\qquad \sum_y\pi(y)=1.$$

1. Subtract a normalisation multiplier `lambda(sum_y pi(y)−1)`.
2. Differentiate in `pi(y)` and set the derivative to zero:

   $$r(y)-\beta[\log(\pi(y)/q(y))+1]-\lambda=0.$$

3. Exponentiate and normalise:

   $$\pi^*(y|x)=\frac{q(y|x)e^{r(x,y)/\beta}}{Z(x)}.$$

4. Rearrange to express reward through an optimal policy:

   $$r(x,y)=\beta\log\frac{\pi^*(y|x)}{q(y|x)}+\beta\log Z(x).$$

5. Now assume Bradley–Terry preferences:
   `P(y_w preferred to y_l | x)=sigmoid(r(x,y_w)−r(x,y_l))`.
   Their shared prompt makes the `beta log Z(x)` terms cancel.
6. Replace the ideal policy with a parameterised policy and minimise preference negative log likelihood:

   $$L_{DPO}=-\log\sigma(z),\quad
   z=\beta\left[(\ell_{\theta,w}-\ell_{q,w})-(\ell_{\theta,l}-\ell_{q,l})\right].$$

Here each `ell` is a **sum of completion-token log probabilities**, including genuine termination tokens.
The implicit reward is already `r_hat=beta(ell_theta−ell_q)`;
putting another beta in front of its difference would count the coefficient twice.
[Rafailov et al.](https://arxiv.org/abs/2305.18290) derive this reward-policy reparameterisation.

The equation is exact for the ideal objective under its assumptions.
Finite preference data, misspecified preference models, LoRA expressivity, and incomplete optimisation
mean a practical DPO run is not guaranteed to equal a PPO run or recover a true reward function.
A deterministic dataset always labelling one response “chosen” can also drive its fitted pair margin
arbitrarily large in a sufficiently flexible unregularised preference fit; beta is not a hard KL bound.

## Worked example: margin, probability, loss, and beta

There are two possible responses. The reference assigns probabilities `(0.5,0.5)`;
the current policy assigns `(0.75,0.25)`, with the first response chosen.
Set `beta=0.5`.

**Predict:** the chosen response improved relative to the reference, so the loss should be below `log 2`.

1. Chosen log-ratio: `log(0.75/0.5)=log 1.5=0.405465`.
2. Rejected log-ratio: `log(0.25/0.5)=log 0.5=−0.693147`.
3. Raw margin: `h=log 1.5−log 0.5=log 3=1.098612`.
4. Scaled margin: `z=beta h=0.549306`.
5. Fitted preference probability: `sigmoid(z)=sqrt(3)/(1+sqrt(3))≈0.633975`.
6. Loss: `log(1+exp(−z))≈0.455746`.
7. Derivative in raw margin: `dL/dh=−beta sigmoid(−beta h)≈−0.183013`.

At the **same policy probabilities**, beta `1` gives preference probability `0.75`
and loss `−log 0.75≈0.287682`. That is a changed loss scale/model, not a measured quality improvement.
In the fixed-reward ideal optimum, increasing beta instead strengthens the reference penalty.
Keep “what this loss does at fixed probabilities” separate from “which policy a regularised objective prefers”.

If current policy equals reference, all relative log probabilities cancel:
`h=z=0`, loss is `log 2≈0.693147`, and `dL/dh=−beta/2`.
This provides a useful first-batch diagnostic when dropout is disabled and reference computation matches.

## Sequence sums, means, and the SFT reference

Suppose a chosen answer has log probability `−6` over `3` tokens, while a rejected answer
has log probability `−8` over `8` tokens.
The chosen answer has higher sequence probability, but mean token log probabilities are
`−2` and `−1`: the ordering reverses after length normalisation.
That is why replacing sums with means silently changes the objective.
Raw answer lengths alone do not determine DPO margins: the reference log probabilities also matter.

For a run starting with `--adapter runs/sft`, the reference is the **initial SFT policy**,
including that adapter. It is not the original checkpoint with the adapter switched off.
The trainer caches completion log probabilities from this fixed starting policy for training and validation pairs.
Those cached values must not change as the trainable adapter updates.
For a fresh zero-initialised adapter, the initial policy equals the underlying base checkpoint.
The [README's DPO command](../README.md) describes the actual saved-adapter workflow.
In [training.py](../llm_lab/training.py), `cache_reference_logprobs` freezes those scores,
`completion_logprobs` sums the masked token terms, and `dpo_losses` applies the logistic loss.
DPO training disables dropout explicitly while keeping training mode for gradient checkpointing.
A frozen reference uses no autograd; the trainable policy must retain autograd.
More generally, `model.eval()` changes layer behaviour but does not itself disable gradients.

## What changes for verifiable online RL?

DPO uses already prepared pairs. In online RL, the policy first samples new answers and a verifier scores them.
For each fixed prompt, sample `G` independent completions and obtain rewards `r_1,...,r_G`.
Let `r_bar` be their mean and `sigma_r²=(1/G)sum_i(r_i−r_bar)²` their population group variance.

RLOO uses

$$b_i=\frac1{G-1}\sum_{j\ne i}r_j,\qquad
A_i^{RLOO}=r_i-b_i=\frac G{G-1}(r_i-\bar r),\qquad G\ge2.$$

Conditional on the prompt, the other completions are independent of completion `i`.
Writing its sequence score as `S_i=grad log pi_theta(y_i|x)`, we get
`E[b_i S_i]=E[b_i]E[S_i]=0`. Thus subtracting this baseline preserves the on-policy reward gradient.
A detached group mean that **includes** `r_i` is still statistically dependent on it:
`E[(r_i−r_bar)S_i]=(1−1/G)E[r_i S_i]`.

GRPO's common group coefficient is

$$A_i^{GRPO}=\frac{r_i-\bar r}{\sigma_r+\epsilon}.$$

Dividing by a random, reward-dependent standard deviation changes weighting across prompts.
Zero-sum coefficients alone do not establish an unbiased gradient or guaranteed variance reduction.
[RLOO](https://arxiv.org/abs/2402.14740) studies leave-one-out REINFORCE;
[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduces GRPO.

### Worked example: two successes and two failures

**Predict:** successful answers get positive coefficients; failed answers get negative ones.
For rewards `(1,1,0,0)`:

1. Group mean is `0.5` and centred rewards are `(0.5,0.5,−0.5,−0.5)`.
2. Standard deviation is `sqrt((4×0.25)/4)=0.5`.
3. A successful answer's peer mean is `1/3`, so its RLOO advantage is `2/3`.
   A failed answer's peer mean is `2/3`, giving `−2/3`.
4. GRPO coefficients are approximately `(1,1,−1,−1)` when epsilon is negligible.
5. For `(1,1,1,1)` or `(0,0,0,0)`, every centred numerator is zero.
   A small epsilon avoids dividing by zero but does not create a reward-learning signal.
6. For `G=1`, RLOO has no peer baseline; zero-centred GRPO gives zero advantage.
   Use another justified estimator or collect more samples, rather than calling that RLOO.

## Keep the old policy and reference policy distinct

| Distribution | Role | When it changes |
|---|---|---|
| `pi_theta` | Current trainable policy | Every optimiser update |
| `pi_old` | Behaviour policy that sampled the rollout | When fresh rollouts are collected |
| `q` | Reference anchoring the desired starting behaviour | Fixed for the run here |

For a sampled token, PPO/GRPO uses ratio
`rho=exp(log pi_theta(token|prefix)−log pi_old(token|prefix))`.
The clipped reward surrogate is `min(rho A, clip(rho,1−clip_eps,1+clip_eps) A)`.
For `A=1`, ratio `1.3`, and clip epsilon `0.2`, the contribution is `min(1.3,1.2)=1.2`.
For `A=−1`, the same ratio gives `min(−1.3,−1.2)=−1.3`: correction is still active.
A reference penalty compares the current policy with `q`, not with the old rollout snapshot.
Its sampling distribution, length normalisation, and gradient convention must be specified separately.

## Reward hacking starts with the verifier contract

For addition, define whether reward requires exactly the correct integer, permits surrounding text,
or accepts the final number in an arbitrary string. These are different tasks.
A policy can learn to print numbers in a rewarded format without computing the correct sum.
Report exact-answer success and format compliance separately; inspect actual generated answers.
Passing one final-answer check also does not validate any preceding reasoning trace.
Review [week 21's lab](../../week-21-reward-hacking-oversight/teaching_guide.md) before expanding the reward.

## Self-checks

<details>
<summary>1. After SFT, can you disable its adapter to obtain the DPO reference?</summary>

Not if the intended reference is the SFT starting policy. Disabling the adapter returns the underlying checkpoint.
Use a frozen copy of the SFT policy or cache its fixed completion log probabilities, as this lab does.

</details>

<details>
<summary>2. Does a rising chosen-minus-rejected margin prove the chosen likelihood increased?</summary>

No. Both likelihoods may decrease while the rejected one decreases faster.
Inspect chosen and rejected log probabilities individually and evaluate generated answers.

</details>

<details>
<summary>3. Does a verifier reward of one for every sampled answer mean GRPO will keep improving them?</summary>

The reward-centred group advantages are zero. The group supplies no within-prompt reward contrast.
A separate KL/entropy term may still update parameters, but that is a different signal.

</details>
