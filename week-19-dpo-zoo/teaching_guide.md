# Week 19 teaching guide: track the margin before comparing the losses

Pair this with [the notes](notes.tex) and [the workbook](workbook.ipynb).
The examples use supplied sequence log probabilities; they are not measured model results.

**Practical companion:** [the LLM lab](../llm-lab/README.md#run-the-full-small-experiment)
trains DPO on a real pretrained model, starting from a saved SFT adapter and caching
the complete initial policy as its fixed reference.

## What you need from week 18

Know the KL-regularised optimum `p_star(y|x) ∝ q(y|x) exp(r(x,y)/beta)`.
Know the Bradley–Terry model `P(w preferred to l)=sigmoid(r_w−r_l)`.
Keep complete-response log probability separate from average token log probability.
The same Greek letter in two algorithms need not have the same scale or operational meaning.

## Analogy: judging progress relative to a starting level

Two musicians compete after training. A judge compares how their odds have changed
relative to their starting recordings.

| Analogy | DPO quantity |
|---|---|
| Starting performance distribution | Frozen reference `q` |
| Current performance distribution | Policy `p_theta` |
| Change relative to the starting odds | `log(p_theta(y)/q(y))` |
| Chosen and rejected recordings of the same piece | Responses `y_w,y_l` to one prompt |
| Difference in relative progress | Raw log-ratio margin `h` |

The analogy explains why absolute likelihood is insufficient: a chosen response
can still be less likely than the rejected one while improving relative to the reference.
It does not explain whether the judge's preference is good; noisy or biased labels remain possible.
Nor does a larger relative margin guarantee the chosen response's probability increased.

## Derivation: DPO without counting beta twice

Fix one prompt `x`, a reference with positive support, and `beta>0`.
Assume the reward-induced partition function is finite.
The ideal derivation uses an expressive policy class and Bradley–Terry preferences
with their noise scale absorbed into the reward.

1. Take logs of the ideal optimum:

   $$\log p^*(y|x)=\log q(y|x)+r(x,y)/\beta-\log Z(x).$$

2. Rearrange and define implicit reward:

   $$r(x,y)=\beta\log[p^*(y|x)/q(y|x)]+\beta\log Z(x),
   \quad \hat r_\theta(x,y)=\beta\log[p_\theta(y|x)/q(y|x)].$$

3. Compare two responses to the **same prompt**. Their `beta log Z(x)` terms cancel.
   Define a raw margin `h` and a scaled margin `z`:

   $$h=(\log p_w-\log q_w)-(\log p_l-\log q_l),\qquad z=\beta h.$$

4. Fit the observed preference:

   $$L_{DPO}=-\log\sigma(z)=\log(1+e^{-z}).$$

5. Differentiate the scalar margin:

   $$\frac{\partial L}{\partial h}=-\beta\sigma(-\beta h).$$

Because `r_hat` already includes `beta`, writing `sigmoid(beta(r_hat_w−r_hat_l))`
would accidentally square that coefficient.
At `h=0`, the gradient magnitude is `beta/2`; a smaller beta does not itself mean faster updates.
[Rafailov et al.](https://arxiv.org/html/2305.18290v3) derive the reward-policy reparameterisation.
It connects ideal objectives; it does not make finite DPO and PPO training trajectories identical.

## Worked example 1: a DPO loss and a likelihood-displacement trap

**Predict.** A policy that improves the chosen response's likelihood relative to the
reference more than the rejected response should have loss below `log 2`.

Let policy log probabilities be `log p_w=−3`, `log p_l=−5`.
Let reference log probabilities both equal `−3.5`; set `beta=0.1`.

1. Relative log probabilities: `ell_w=0.5`, `ell_l=−1.5`.
2. Raw margin: `h=0.5−(−1.5)=2`.
3. Implicit rewards: `r_hat_w=0.05`, `r_hat_l=−0.15`; scaled margin `z=0.2`.
4. Preference probability: `sigmoid(0.2)=0.549834`.
5. Loss: `log(1+exp(−0.2))=0.598139`, below `0.693147`.
6. Margin derivative: `−0.1 sigmoid(−0.2)=−0.0450166`.

**Edge case.** Start from the reference and instead move policy log probabilities
to `−4.5` for the chosen and `−6.5` for the rejected response.
Now relative log probabilities are `−1` and `−3`, so `h=2` and the loss is **identical**.
Yet the chosen response's probability fell from `exp(−3.5)=0.0301974`
to `exp(−4.5)=0.0111090`.
The remaining probability mass can move to other responses.
Pairwise margin improvement alone is not evidence of improving chosen-response likelihood.

## Worked example 2: DPO, IPO, and SimPO measure different things

Use the original pair `log p_w=−3`, `log p_l=−5`, with lengths `10` and `30`.
The reference remains `−3.5` for each response.

**Predict.** The chosen response has higher sequence probability, but the rejected
response can have higher average token probability.

1. Mean token log probabilities: `−3/10=−0.3`, `−5/30=−1/6≈−0.166667`.
2. For SimPO `beta=2`, margin offset `gamma=1`:
   `z_simpo=2(−0.3+1/6)−1=−1.266667`.
3. Its preference probability is `0.219828`; its loss is `1.514908`.
4. DPO with `beta=0.1` still gives `h=2`, `z=0.2`, loss `0.598139`.
5. The project's IPO with `beta=0.1` targets raw margin `1/(2 beta)=5`:
   loss `(2−5)²=9`; derivative in `h` is `2(2−5)=−6`.
6. If the raw margin reaches `7`, IPO loss is `(7−5)²=4` and derivative `+4`:
   gradient descent now pushes the margin **down** toward its finite target.

**Interpret.** Loss magnitudes are not comparable measures of response quality.
The variants differ in scaling, normalisation, and objective.
An average token log probability is the logarithm of a geometric mean;
it is not the normalised probability of a full response.
A finite scalar IPO target does not imply every pair in a shared model can attain it:
contradictory or cyclic preferences may force compromises.
[SimPO](https://arxiv.org/abs/2405.14734) explicitly uses length-normalised scores;
[IPO](https://arxiv.org/abs/2310.12036) studies a different preference objective.

## Bridge to the actual project

- [project/losses.py](project/losses.py): `dpo_loss` subtracts reference log probabilities
  before multiplying the pairwise difference by `beta`.
- `ipo_loss` uses the raw log-ratio difference; `simpo_loss` divides by lengths.
- `implicit_reward_margin` returns the **scaled** margin, not `h`.
- `_log1mexp` supports the odds transformation in `orpo_loss`; its input must be nonpositive.
- [project/solutions.py](project/solutions.py) isolates the implicit reward, stable
  logarithm, and loss variants before the workbook's bake-off.

Read the caller as well as the loss: whether ORPO receives summed or averaged
response log probabilities changes the convention. The toy module documents summed inputs;
do not silently equate every toy variant with a published recipe.
Keep the reference frozen and check completion masking before diagnosing optimiser behaviour.
Report held-out preferences, length distributions, and chosen/rejected likelihoods separately.

## Retrieve and transfer

<details>
<summary>1. What is DPO loss when the current policy equals the reference?</summary>

Both relative log probabilities are zero, so `h=z=0` and loss is `log 2≈0.693147`,
regardless of the individual response probabilities or beta.

</details>

<details>
<summary>2. Why must the pair share a prompt for partition-function cancellation?</summary>

For different prompts the terms are `beta log Z(x_1)` and `beta log Z(x_2)`.
They need not match. Within-prompt comparisons do not identify an absolute cross-prompt reward scale.

</details>

<details>
<summary>3. Does multiplying every completion length by two leave SimPO unchanged?</summary>

Not if log probabilities stay fixed: each mean log probability halves, shrinking their difference.
If both the summed log probabilities and lengths double, the means and SimPO margin stay fixed.
This is an algebraic invariance, not a guarantee about duplicating real text under an autoregressive model.

</details>
