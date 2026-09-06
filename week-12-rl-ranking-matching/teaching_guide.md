# Week 12 teaching guide — distinguish relevance, exposure, and selection

Use this guide alongside exercises 4 and 6–9, then apply the distinctions in the recommendation memo.

## Connect to what you know

Week 11 evaluated a chosen action using its logging propensity.
Ranking adds a second selection mechanism: an item must be displayed and examined before it can be clicked.
Supervised retrieval estimates similarity; bandit reranking adapts decisions using feedback.
Neither a large similarity score nor a high rank automatically represents a calibrated probability.

## Analogy: books on a library display

The librarian's catalog search maps to retrieval; arranging a short display maps to reranking.
Eye-level shelves receive more attention: that is position-dependent examination.
A book's appeal maps to relevance, while checkout is the observed click/outcome.
A quiet lower shelf can hide a useful book even when its relevance is high.

**Limit:** real users compare items and change attention after earlier clicks.
The simple position-bias model assumes an item's examination probability depends only on its position.
It does not model competition, trust in the ranking, or two-sided capacity constraints.
A candidate/job assignment is more than a set of independent ranked lists.

## Derivation: change examination position, not item identity

For one fixed query, let item $i$ have relevance probability $q_i$.
Let $e_k>0$ be examination probability at position $k$.
Assume examination and relevance produce click $C_{i,k}$ with
$\mathbb E[C_{i,k}]=e_kq_i$.
Also assume the target uses only items covered by the logged display.

1. The target expected click sum is $V=\sum_i e_{k'(i)}q_i$.
2. A logged click at position $k(i)$ estimates relevance by $C_{i,k(i)}/e_{k(i)}$.
3. Place that relevance estimate at its target position:

   $$\widehat V=\sum_i C_{i,k(i)}\frac{e_{k'(i)}}{e_{k(i)}}.$$

4. Take expectation term by term:

   $$\mathbb E\left[C_{i,k(i)}\frac{e_{k'(i)}}{e_{k(i)}}\right]
   =e_{k(i)}q_i\frac{e_{k'(i)}}{e_{k(i)}}=e_{k'(i)}q_i.$$

5. Sum to recover $V$ under this click model.

This is a model-specific estimator for a covered set of items.
If the logging policy sometimes omits a target item, item inclusion requires its own coverage/propensity reasoning.
Correcting examination alone does not evaluate arbitrary unseen slates.
See [Joachims, Swaminathan, and Schnabel](https://arxiv.org/abs/1608.04468) for counterfactual learning to rank.

## Worked example 1 — swapping two items

**Predict:** should promoting a relevant but poorly exposed item increase expected clicks?
Let relevance probabilities be $q_A=0.2$, $q_B=0.8$ and examination $(1,0.5)$.
The logger displays (A,B); the target displays (B,A).

**Calculate:** logger expected clicks are $1(0.2)+0.5(0.8)=0.6$.
Target expected clicks are $1(0.8)+0.5(0.2)=0.9$.
Suppose one logged display produces clicks $(0,1)$.
A's weight is $0.5/1=0.5$, B's weight is $1/0.5=2$.
The estimate is $0(0.5)+1(2)=2$.
Averaging in expectation gives $0.2(0.5)+0.4(2)=0.9$.

**Interpret:** a single estimate can be much larger than its expectation.
Reweighting corrects a sampling mechanism in expectation, not every noisy display.
If examination at B's logged position were zero, this calculation could not identify its relevance.

## Worked example 2 — equal representation is not equal selection rates

**Predict:** does selecting one person from each group guarantee demographic parity?
A candidate pool has two F candidates and eight M candidates.
A top-two list contains one F and one M.

**Calculate:** representation shares are $1/2$ and $1/2$, so their gap is zero.
Selection rates conditional on group are $1/2=0.5$ and $1/8=0.125$.
The selection-rate demographic-parity gap is $|0.5-0.125|=0.375$.

Now suppose the qualified pool contains one F and four M candidates,
and both selected candidates are qualified.
Equal-opportunity rates are $1/1=1$ and $1/4=0.25$, a gap of $0.75$.

**Interpret:** these metrics use different denominators and answer different questions.
The project's legacy `demographic_parity_gap` name actually computes representation-share disparity.
That implementation is not the conditional selection-rate definition used in much of the literature.
State the formula before comparing a value against any target.
None of these scalar summaries alone establishes that a system is fair.

## Worked example 3 — a ranking can improve while calibration is unchanged

**Predict:** are rank ordering and calibrated probabilities the same property?
Take relevance labels $(3,1,0)$ and predicted order $(1,3,0)$.
Use gains $2^{rel}-1$ and discounts $1/\log_2(k+1)$.

**Calculate:** DCG $=1+7/\log_2(3)+0=5.416508$.
Ideal DCG $=7+1/\log_2(3)=7.630930$.
NDCG $=5.416508/7.630930=0.709810$.
Swapping the first two items makes NDCG exactly 1.
A strictly increasing transformation of all scores preserves either ranking and its NDCG.

For probabilities $(0.9,0.9)$ and binary outcomes $(1,0)$,
Brier score is $[(0.9-1)^2+(0.9-0)^2]/2=0.41$.
Replacing both probabilities by $0.5$ gives Brier score $0.25$.
Neither choice distinguishes the two items for ranking.

**Interpret:** probability accuracy and ordering quality need separate evaluation.
A two-example Brier calculation is illustrative; estimating calibration needs more held-out outcomes.

## Code bridge

- [cltr.py](project/cltr.py): `ips_ltr_value` uses target examination divided by logged examination.
  `expected_relevance` is the synthetic expected-click reference despite its shorter name.
- [train.py](project/train.py): `dcg` and `ndcg_at_k` implement exponential gains;
  check the exact gain convention before comparing a hand calculation to another library.
- [fairness.py](project/fairness.py): `demographic_parity_gap` is the representation metric above.
  `equalised_opportunity_gap` conditions on synthetic qualification; inspect empty-group handling.
- `exposure_by_group` averages exposure per candidate and across jobs over the entire ranking.
  It is different from total group exposure or exposure limited to the displayed top-k.
- [calibration.py](project/calibration.py): fit calibrators on a calibration split and evaluate on held-out data.
  Neither a particular calibrator nor a fixed training-set size guarantees lower held-out ECE.
- [two_tower.py](project/two_tower.py): `TwoTower.score` produces a similarity score.
  Training with in-batch negatives is not by itself probability calibration.

For the memo, report one concrete metric with its numerator, denominator, cutoff, and data split.
Replace “the model is fair” with the actual audited property and which groups/jobs the evidence covers.
This is an interpretation exercise; the synthetic examples do not settle real deployment policy.

## Retrieval and transfer

<details>
<summary>1. Can an examination correction recover clicks for an item never displayed?</summary>

No. It only corrects observed exposure under its model. Evaluation of omitted items needs support from randomized inclusion or explicit extrapolation assumptions.
</details>

<details>
<summary>2. Why might an overall exposure gap hide a problem on individual jobs?</summary>

Positive and negative disparities can cancel when aggregated. Report the per-job distribution and empty/rare-group cases as well as the overall mean.
</details>

<details>
<summary>3. A monotone calibrator leaves NDCG unchanged. Has it failed?</summary>

No. Its job is to improve probability interpretation. A strictly increasing transformation preserves rank, so unchanged NDCG is expected.
</details>

Arithmetic checks: click weights, representation/selection denominators, NDCG, and Brier scores
were independently recomputed in Python. The guides distinguish synthetic ground truth from logged evidence.
