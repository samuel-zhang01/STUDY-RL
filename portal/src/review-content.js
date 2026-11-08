// Questions are grounded in each teaching guide and its bundled project.
// They are retrieval practice, not a certification or a replacement for coding.
const REPO = 'https://github.com/samuel-zhang01/STUDY-RL';
const rows = [];
let current;
const sectionId = value => value.toLowerCase().replace(/[^\p{L}\p{N}_\- ]/gu,'').replace(/ /g,'-');
function week(number, folder) { current = {number, folder}; }
function q(key, kind, prompt, options, section, task, file = 'project/solutions.py', repairWeek = current.number) {
  const week = String(current.number).padStart(2, '0');
  // Rotate whole answer/feedback pairs deterministically: the correct answer is
  // authored first, but is not always displayed first. IDs survive reordering.
  const offset = (current.number + ['concept','calculate','transfer'].indexOf(kind)) % options.length;
  const rotated = options.slice(offset).concat(options.slice(0, offset));
  rows.push({id:`w${week}-${key}`, week, kind, prompt,
    choices:rotated.map(x => x[0]), feedback:rotated.map(x => x[1]),
    correctIndex:(options.length-offset)%options.length,
    repair:{route:`#/lesson/${String(repairWeek).padStart(2,'0')}?view=guide&section=${encodeURIComponent(sectionId(section))}`, label:`Revisit week ${String(repairWeek).padStart(2,'0')}: ${section}`},
    sourceSection:section,
    handoff:{route:`#/lesson/${week}?view=workbook`,label:`Open week ${week} workbook`,
      source:`${REPO}/blob/main/${current.folder}/${file}`, path:`${current.folder}/${file}`, task},
    notebook:`${current.folder}/workbook.ipynb`});
}

week(1, 'week-01-mdps-bandits');
q('reward-regret','concept','An arm with mean reward 0.5 pays 1. The best arm has mean 0.8. What did this pull contribute to pseudo-regret?',[
 ['0.3, even though the observed reward was 1.','Pseudo-regret uses the means: 0.8 − 0.5 = 0.3. A lucky outcome does not erase expected opportunity cost.'],
 ['−0.2, because the observed reward exceeded 0.8.','That subtracts a realised reward from a mean. Pseudo-regret compares the best mean with the selected arm’s mean.'],
 ['0, because every successful pull is optimal.','Success is one random observation. An inferior arm can succeed while the optimal arm can fail.']
], 'Example 1: update, regret, and one lucky observation', 'Compare env.regret with the realised reward sum on the same seeded trajectory.', 'project/env.py');
q('arm-count','calculate','An arm has been pulled twice, with mean 0.5. On global round 20 you pull it again and observe 1. What is its new sample mean?',[
 ['2/3','Its old reward sum is 2 × 0.5 = 1. Add 1 and divide by this arm’s new count, 3.'],
 ['0.525','Using 1/20 weights observations by global rounds. Other arms’ observations do not belong in this arm’s sample mean.'],
 ['0.75','A step size of 1/2 treats the new observation as half the history. It is the third observation, so its weight is 1/3.']
], 'Derivation: from an average to an online update', 'Run exercise_1_incremental_mean on [1, 0, 1], then implement the update yourself.');
q('capacity-state','transfer','A delivery route is initially a bandit arm. Choosing it now consumes capacity needed by later deliveries. What model change addresses this consequence?',[
 ['Include remaining capacity in a state and model how actions change it.','Capacity links current actions to future opportunities. A state transition exposes that delayed consequence to the objective.'],
 ['Increase epsilon; the stationary bandit model now captures the consequence.','More exploration changes sampling, not the model’s missing state or delayed reward.'],
 ['Use total deliveries instead of arm count in the running mean.','Changing the denominator damages the reward estimate and does not represent capacity dynamics.']
], 'Analogy: choosing a lunch stall', 'Extend a small bandit fixture with a remaining-capacity state; write two transitions before training anything.', 'project/env.py');

week(2, 'week-02-bellman-dp');
q('evaluation-improvement','concept','What distinguishes a Bellman policy-evaluation backup from a Bellman optimality backup?',[
 ['Evaluation averages over the fixed policy; optimality maximises over actions.','Both average over transition outcomes. The action average versus maximum is the difference.'],
 ['Evaluation maximises over random next-state outcomes.','The agent does not choose which random outcome occurs. Both backups average using the transition probabilities.'],
 ['Evaluation ignores all future rewards.','It includes discounted continuation value. Fixing the policy does not remove the future.']
], 'Derivation: why the Bellman equation is true', 'Compare exercise_1_bellman_operator with policy evaluation in exercise_6_policy_evaluation_closed_form.');
q('continuing-loop','calculate','A gives reward 1 then moves to B. B gives reward 2 and stays at B forever. With gamma = 0.5, what are (V(A), V(B))?',[
 ['(3, 4)','V(B) = 2 + 0.5V(B) = 4; V(A) = 1 + 0.5 × 4 = 3. B is continuing, not terminal.'],
 ['(2, 2)','Those values fit a different task where B pays once and terminates. Here B keeps paying.'],
 ['(1, 2)','This is only the first synchronous sweep from zero. It omits all later rewards.']
], 'Example 1: solve a two-state policy by hand', 'Construct this two-state system and compare a linear solve with repeated synchronous backups.');
q('stable-policy','transfer','Modified policy iteration stops after one evaluation sweep because its chosen action did not change. What additional evidence is needed for accurate values?',[
 ['Check a suitably small Bellman residual, interpreted relative to 1 − gamma.','A policy can stay unchanged while its values are still far from their fixed point. The residual bounds value error under the stated discounted-MDP assumptions.'],
 ['No other check: unchanged actions imply exact values.','A one-action reward-1 loop never changes policy, even while its value moves from 0 toward 1/(1 − gamma).'],
 ['Check whether a random rollout happened to obtain high return.','A single sampled return does not certify that the value vector satisfies its Bellman equation.']
], 'Derivation: why repeated backups converge', 'Inspect exercise_7_gpi and test a one-state, one-action reward-1 loop.');

week(3, 'week-03-monte-carlo');
q('first-visit-order','concept','Why can a backward return-computation loop accidentally implement last-visit rather than first-visit Monte Carlo?',[
 ['The first unseen pair encountered backward is its last occurrence in forward time.','Return computation and visit selection are separate: compute returns backward, then identify earliest occurrences in chronological order.'],
 ['Discounted returns must always be computed forward.','The backward recurrence is efficient and correct. The error is which occurrence supplies the sample.'],
 ['First-visit Monte Carlo must discard all episodes with repeated states.','It keeps the first occurrence in each episode; repetition is allowed.']
], 'Example 1: backwards computation is not backwards selection', 'Inspect the two passes in exercise_1_first_visit_returns.');
q('first-return','calculate','An episode is (A,x,1), (B,y,2), (A,x,3), then termination. With gamma = 0.5, which return does first-visit MC retain for (A,x)?',[
 ['2.75','The first return is 1 + 0.5 × 2 + 0.25 × 3 = 2.75.'],
 ['3','That is the return at the last A visit, not the first one.'],
 ['2.875','That averages the two A returns, 2.75 and 3. It is the every-visit estimate for this single episode.']
], 'Example 1: backwards computation is not backwards selection', 'Reproduce the repeated-state episode in exercise_1_first_visit_returns.');
q('trajectory-support','transfer','A target policy sometimes takes an action the behaviour policy never takes. Can arbitrarily many behaviour episodes make ordinary trajectory importance sampling identify that target value?',[
 ['No; the target uses trajectories with zero behaviour support.','A likelihood ratio cannot recover unobserved outcomes where its denominator is zero. Change data collection or restrict the target policy.'],
 ['Yes; enough episodes remove every importance-sampling problem.','More samples reduce some sampling error but never observe a zero-probability event under the unchanged logger.'],
 ['Yes; replacing a zero probability with 0.001 creates overlap.','The replacement changes a number, not the data-generating process. It invents support rather than supplying evidence.']
], 'Derivation: importance sampling changes the sampling distribution', 'Add a coverage check before exercise_4_ordinary_is computes trajectory ratios.');

week(4, 'week-04-td-sarsa-q');
q('on-off-policy','concept','SARSA samples a poor exploratory next action. Q-learning sees the same transition. Which continuation does each use?',[
 ['SARSA uses that sampled action’s Q; Q-learning uses the largest next-action Q.','SARSA follows its realised behaviour action. The Q-learning target is greedy, even when behaviour explored.'],
 ['Both must use the sampled action because it occurred in the data.','That describes SARSA. Q-learning learns from sampled transitions while using a greedy continuation target.'],
 ['Q-learning uses the actual reward from the next action.','Its one-step target bootstraps from an estimated maximum Q; it need not wait for another reward.']
], 'Example 1: the same transition, three different lessons', 'Compare the targets in workbook exercises 3–5 on one fixed transition.');
q('terminal-target','calculate','Old V(s) = 2, reward = 5, gamma = 0.9, alpha = 0.5. The next state is truly terminal but its stored value is 100. What is the updated V(s)?',[
 ['3.5','True termination removes continuation: target = 5, so 2 + 0.5(5 − 2) = 3.5.'],
 ['48.5','That incorrectly bootstraps from the stored 100 after true termination.'],
 ['5','That replaces the estimate with the target, equivalent to alpha = 1 rather than 0.5.']
], 'Example 2: termination is an objective boundary', 'Test the same transition with terminal and artificial-cutoff flags separately.');
q('cutoff-bootstrap','transfer','A continuing task stops collection after 100 steps only to make a training batch. What should the last target usually do?',[
 ['Bootstrap from the final pre-reset observation and stop the trace from crossing into a reset episode.','Collection stopped, but the task’s future rewards did not disappear. Keep the appropriate continuation estimate while respecting sequence boundaries.'],
 ['Zero the continuation because any stopped collector implies task termination.','This changes the continuing objective by treating a sampling boundary as the end of all future return.'],
 ['Bootstrap from the first observation after resetting the environment.','That observation belongs to a different episode and is not the successor of the final recorded action.']
], 'Episode boundaries in the project', 'Trace the collector’s final observation and terminal/truncation masks before changing a learner.');

week(5, 'week-05-dqn');
q('detached-target','concept','In the standard DQN semi-gradient update, where should the loss gradient flow?',[
 ['Through the online Q prediction for the recorded action, while the target is held fixed.','The target network supplies a label. Differentiating through that label changes the mathematical update.'],
 ['Only through the target-network maximum.','That would move the answer sheet instead of fitting the recorded action’s online prediction.'],
 ['Equally through every online action value in the row.','Gather selects the recorded action’s output. Other actions are not supervised with this same transition label.']
], 'Derivation: what receives the gradient', 'Inspect gather and torch.no_grad in DQNAgent.learn.', 'project/agent.py');
q('linear-gradient','calculate','Let Q = theta × x, x = 2, theta = 1 and fixed target y = 3. Minimise half squared error with learning rate 0.1. What is the new theta?',[
 ['1.2','The gradient is (Q − y)x = (2 − 3) × 2 = −2. Descent gives 1 − 0.1(−2) = 1.2.'],
 ['0.8','That follows the gradient rather than descending it; it moves the prediction away from the larger target.'],
 ['1.1','That omits the derivative dQ/dtheta = x = 2. The parameter gradient differs from the output error.']
], 'Example 2: a tiny linear network exposes the sign', 'Check the hand gradient with a one-parameter tensor before running exercise_8_smoke_dqn.');
q('batch-shape','transfer','A two-row replay batch gathers predictions (3,4), while targets accidentally have shape (2,1). Why inspect this before trusting a falling loss?',[
 ['Broadcasting can compare every prediction with both targets, producing a 2×2 loss.','A loss can decrease for the wrong pairings. Assert compatible per-row shapes before reducing.'],
 ['Broadcasting always pairs rows correctly when both contain two numbers.','Shapes, not the number of entries alone, control broadcasting. (2,) with (2,1) expands to (2,2).'],
 ['A target network guarantees shape correctness.','A target network delays labels; it does not validate how tensors align.']
], 'Notebook investigation: make a batch-shape bug visible', 'Run exercise_1_gather_q and exercise_2_dqn_target, printing shapes before subtraction.');

week(6, 'week-06-rainbow-bbf');
q('double-selection','concept','What is the central separation in Double DQN?',[
 ['The online network selects the next action; the target network evaluates that selected action.','Separating selection and evaluation changes the maximum’s estimation bias. It does not eliminate all estimation error.'],
 ['Average two independent maxima.','Averaging maxima is not Double DQN’s online-selection/target-evaluation rule.'],
 ['Represent the return distribution on fixed atoms.','That describes a distributional representation such as C51; it is a different Rainbow component.']
], 'Example 3: Double and Dueling are different operations', 'Find the action-selection and evaluation tensors in RainbowAgent.learn.', 'project/agent.py');
q('per-correction','calculate','A two-row buffer has sampling probabilities (0.25,0.75) and row gradients (2,6). With exact unnormalised importance weights 1/(2P), what is the expected corrected gradient?',[
 ['4','Weights are (2,2/3): 0.25×2×2 + 0.75×(2/3)×6 = 4, matching uniform replay.'],
 ['5','That is the unweighted prioritised expectation, 0.25×2 + 0.75×6.'],
 ['2','That follows after additionally dividing all weights by their full-buffer maximum 2. It rescales the expectation.']
], 'Example 1: biased sampling, exact correction, and a normalisation trap', 'Compare exact PER weights with full-buffer and minibatch-maximum normalisation.');
q('distribution-uncertainty','transfer','A C51 agent’s return distribution is broad. What can you conclude from width alone?',[
 ['Returns vary under the represented outcome distribution; width alone is not a confidence interval for the estimated mean.','Outcome variability and uncertainty from limited data are different. A well-known stochastic outcome can remain broad.'],
 ['The agent necessarily has too little data.','Even unlimited data can reveal a genuinely broad return distribution.'],
 ['There is a 95% guarantee that the true mean lies within the atom range.','The fixed support is a modelling choice, often with clipping. It does not create a statistical confidence guarantee.']
], 'Notebook investigation: distinguish a distribution from a confidence interval', 'Inspect the C51 projection exercise and change its support without changing the input outcomes.');

week(7, 'week-07-policy-gradients');
q('baseline-mean','concept','Why can a state-only baseline preserve the expected REINFORCE gradient?',[
 ['For a fixed state, the policy score has mean zero over actions.','E[b(s)∇log π(a|s)] = b(s)∇Σaπ(a|s) = 0 when the baseline is held fixed with respect to the sampled action.'],
 ['Every baseline makes every sampled gradient smaller.','Individual gradients can grow, and even variance can increase. Preserving the mean is a separate property.'],
 ['Subtracting any action-dependent reward is harmless.','An action-dependent baseline generally correlates with the score and can change the expected gradient unless corrected.']
], 'Derivation: why a baseline preserves the mean', 'Enumerate the two-action estimator with and without a state baseline.');
q('bernoulli-score','calculate','A one-step Bernoulli policy has p = 0.2, reward r(a) = a and logit theta. What is d E[r] / d theta?',[
 ['0.16','E[r] = sigmoid(theta), whose derivative is p(1 − p) = 0.2×0.8 = 0.16.'],
 ['0.8','That is the score and sampled gradient conditional on action 1, before averaging over its probability 0.2.'],
 ['0.2','That is the expected reward, not its derivative with respect to the logit.']
], 'Example 1: two actions let us enumerate the exact gradient', 'Enumerate both actions and probability-weight their score-function gradients.');
q('baseline-variance','transfer','Your value baseline preserves the estimated gradient mean but increases its measured variance. Is that mathematically impossible?',[
 ['No; the variance-minimising scalar baseline weights returns by squared score magnitude.','The ordinary expected return need not equal E[||score||²G]/E[||score||²]. Check the gradient estimator, not just return variance.'],
 ['Yes; any unbiased baseline must reduce variance.','Unbiasedness fixes an expectation, not the estimator’s spread.'],
 ['Yes; a value baseline makes every policy-gradient sample identical.','A baseline does not generally remove action or trajectory randomness.']
], 'Example 2: the value baseline can increase variance', 'Reproduce the guide’s two-action counterexample and compare gradient variances.');

week(8, 'week-08-a2c-gae-trpo');
q('lambda-one','concept','At lambda = 1, a GAE rollout ends at an artificial batch cutoff. Does its return target necessarily become a pure completed-episode Monte Carlo return?',[
 ['No; the final continuation value survives even though intermediate value terms telescope.','Lambda = 1 cancels intermediate bootstrap terms, not the unfinished segment’s final bootstrap.'],
 ['Yes; lambda = 1 removes every value prediction.','This is true only when the remaining endpoint contribution is zero, such as a true terminal boundary.'],
 ['No; lambda = 1 removes all rewards and leaves only values.','The rewards remain. Telescoping leaves their discounted sum plus the endpoint bootstrap.']
], 'Example 2: lambda one does not remove an unfinished rollout’s bootstrap', 'Run compute_gae with the same rewards under terminal and truncated endings.');
q('gae-residuals','calculate','Two TD residuals are (1,2). Gamma = 0.9 and lambda = 0.5, with the trace ending after the second residual. What is the first GAE advantage?',[
 ['1.9','A₀ = δ₀ + gamma×lambda×δ₁ = 1 + 0.9×0.5×2 = 1.9.'],
 ['2.8','That uses gamma but omits lambda, corresponding to lambda = 1 for these residuals.'],
 ['1','That uses only the current residual, corresponding to lambda = 0.']
], 'Derivation: the telescoping identity behind GAE', 'Compare compute_gae at lambda 0, 0.5 and 1, with identical residuals and boundary masks.');
q('trpo-guarantee','transfer','A practical TRPO implementation accepts a step using a finite-batch average KL estimate. What should you claim?',[
 ['It passes that implementation’s sampled acceptance checks; an exact monotonic-return theorem needs stronger assumptions.','Finite samples, approximate optimisation and average rather than worst-case constraints separate practical checks from the ideal guarantee.'],
 ['Every state’s KL is bounded by the sampled average.','A small average allows larger deviations in some states and says less about unobserved states.'],
 ['True expected return is guaranteed to increase after every accepted step.','The practical surrogate and sampled checks alone do not establish the ideal theorem’s premises.']
], 'Derivation: the natural-gradient trust-region step', 'Inspect the line-search acceptance criteria and distinguish measured KL from population claims.');

week(9, 'week-09-ppo');
q('negative-advantage','concept','A sampled action has negative advantage and its probability ratio rises above 1 + epsilon. Does PPO clipping necessarily remove its actor gradient?',[
 ['No; the harmful probability increase can remain on the unclipped, decreasing objective branch.','The minimum keeps penalties for harmful moves. Clipping removes further improvement incentives in the favourable direction.'],
 ['Yes; every ratio outside the interval gets zero gradient.','The sign of the advantage determines which branch wins. Being outside alone is insufficient.'],
 ['Yes; negative advantages are discarded from PPO batches.','Negative advantages teach the policy to reduce probability for worse-than-baseline actions.']
], 'Derivation: keep the sign visible', 'Use ppo_clip_loss with both advantage signs at the same ratio.');
q('ppo-loss-sign','calculate','Ratio = 1.5, advantage = −2 and epsilon = 0.2. What is this sample’s minimised PPO actor loss, excluding entropy and value terms?',[
 ['3','Raw surrogate = −3; clipped candidate = −2.4; min = −3. The minimised loss negates it, giving 3.'],
 ['2.4','That chooses the clipped candidate without taking the required minimum.'],
 ['−3','That is the surrogate to maximise, before negating it for a minimiser.']
], 'Worked example 1 — the same ratio, opposite advantages', 'Compare ppo_clip_loss with the browser’s PPO geometry plot at this exact point.');
q('clip-not-bound','transfer','After many PPO epochs, some action ratios are 1.8 with epsilon = 0.2. Does that alone prove the clipping code failed?',[
 ['No; the clipped surrogate does not impose a hard bound on every ratio or on KL.','Shared parameters and other samples can move ratios beyond the interval. Inspect the objective, measured KL and stopping rule.'],
 ['Yes; clipping projects the policy so all ratios lie between 0.8 and 1.2.','PPO clips an objective candidate, not the policy probabilities themselves.'],
 ['No; ratios are always unrelated to policy changes.','Ratios do quantify sampled-action probability changes. They are informative diagnostics, just not hard constraints.']
], 'Retrieval and transfer', 'Log ratio distribution and approximate KL during PPOAgent.update_minibatch.', 'project/agent.py');

week(10, 'week-10-sac-td3');
q('temperature-scale','concept','In a finite-action maximum-entropy objective, both Q and temperature alpha double. What happens to the optimal softmax probabilities?',[
 ['They stay the same because Q/alpha stays the same.','Temperature is relative to the scale of value. exp(Q/alpha) has unchanged ratios.'],
 ['They always become uniform because temperature increased.','That would ignore the simultaneous doubling of Q. Both scales matter.'],
 ['They become deterministic because Q increased.','That would ignore the simultaneous doubling of temperature.']
], 'Worked example 1 — temperature is a scale', 'Calculate softmax probabilities for Q = (0, log 3), alpha = 1, then double both.');
q('soft-terminal','calculate','SAC uses r = 1, gamma = 0.9, next critics (4,6), log pi = −2 and alpha = 0.2. What is the target if the transition truly terminates?',[
 ['1','Termination masks the entire continuation, including its entropy term. Only the observed reward remains.'],
 ['4.96','That is the nonterminal target: 1 + 0.9[4 − 0.2(−2)].'],
 ['1.4','An entropy term is not added after true termination merely because a next-policy density can be computed.']
], 'Worked example 3 — terminal targets and actor-gradient sign', 'Test soft_bellman_target with identical inputs and terminal flags 0 versus 1.');
q('action-rescale','transfer','A one-dimensional continuous action is rescaled from a to a′ = 2a. How must its log density change at corresponding points?',[
 ['Subtract log 2 from the old log density.','Probability mass is preserved while the interval doubles: p(a′) = p(a)/2. This is distinct from the tanh Jacobian.'],
 ['Add log 2 because the action is larger.','Density is per unit length. Stretching the coordinate reduces density at corresponding points.'],
 ['Leave it unchanged because the same random sample was used.','The sample correspondence is unchanged, but the coordinate measure changed. SAC entropy depends on that density.']
], 'Derivation bridge — squashing changes density', 'Add an action-scale Jacobian check beside tanh_gaussian_log_prob.');

week(11, 'week-11-contextual-bandits-ope');
q('ope-overlap','concept','The logger always chooses A, while the target always chooses B. What does a zero SNIPS denominator indicate?',[
 ['The evaluation is unsupported; no observed row represents the target’s action.','The target outcomes are not identified from these logs. A denominator floor cannot create them.'],
 ['The target’s true reward is zero.','No evidence for B is not evidence of zero reward from B.'],
 ['The reward model is necessarily perfect.','The overlap failure concerns observed actions, regardless of a model’s apparent fit to A.']
], 'Worked example 3 — an impossible evaluation', 'Construct this unsupported logger/target pair before calling snips.', 'project/ope.py');
q('ips-range','calculate','Two logged rewards are both 1, with importance weights 2/3 and 2. What are (IPS, SNIPS)?',[
 ['(4/3, 1)','IPS divides weighted reward 8/3 by two rows. SNIPS divides by weight sum 8/3. An individual IPS estimate can exceed the reward range.'],
 ['(1, 1)','Only SNIPS is a normalised weighted average here. Clipping IPS to 1 would change the estimator.'],
 ['(8/3, 4/3)','The first value is the weighted reward sum, before dividing by sample count. SNIPS uses the weight sum, not the row count.']
], 'Worked example 2 — normalization changes the answer', 'Pass this two-row fixture through ips and snips, retaining their distinct denominators.', 'project/ope.py');
q('dr-limits','transfer','A doubly robust estimate uses a misspecified reward model but correctly recorded propensities. Under overlap and the usual fixed-target evaluation assumptions, what follows?',[
 ['The expected residual correction can cancel reward-model error; finite-sample estimates can still be noisy.','Correct weighting supplies one route to consistency/unbiasedness under appropriate fitting and sampling conditions. It is not a finite-sample accuracy guarantee.'],
 ['The estimate must be pessimistic because the reward model is wrong.','There is no universal bias direction. The correction is precisely what can remove model error in expectation.'],
 ['Overlap no longer matters because the estimator is doubly robust.','Double robustness does not recover rewards for actions absent from the logger’s support.']
], 'Derivation: IPS and the DR correction', 'Inspect doubly_robust and fit its reward model separately from evaluation rows.', 'project/ope.py');

week(12, 'week-12-rl-ranking-matching');
q('click-relevance','concept','Under the simple click model P(click) = P(examination) × relevance, what makes raw click rate a misleading relevance comparison across positions?',[
 ['Different positions can have different examination probabilities.','A low click rate may reflect low exposure rather than low relevance. The correction requires assumptions about the examination mechanism.'],
 ['Clicks contain no information about relevance under any conditions.','With positive, appropriately modelled examination, clicks do provide information. They are not automatically unbiased relevance labels.'],
 ['Sorting items by predicted score guarantees equal examination.','Ordering is exactly what can change exposure. A score does not equalise position bias.']
], 'Derivation: change examination position, not item identity', 'Read the target-examination/logged-examination ratio in ips_ltr_value.', 'project/cltr.py');
q('selection-denominator','calculate','A pool has 2 F and 8 M candidates. A shortlist selects 1 from each group. What is the absolute gap in group-conditional selection rates?',[
 ['0.375','F rate = 1/2; M rate = 1/8. Their absolute difference is 0.5 − 0.125 = 0.375.'],
 ['0','Zero is the representation-share gap within the two selected candidates, using a different denominator.'],
 ['0.6','That is the eligible-pool share difference, 0.8 − 0.2. It is not the selection-rate gap.']
], 'Worked example 2 — equal representation is not equal selection rates', 'Write eligible counts beside selected counts before interpreting the fairness helper’s metric.', 'project/fairness.py');
q('zero-examination','transfer','An item is always logged in a position with zero examination probability. Can reweighting its logged non-clicks identify its relevance at the top position?',[
 ['No; the logger supplied no examined outcomes for that item in this model.','A zero examination probability is a support failure. Additional exposure or stronger external assumptions are needed.'],
 ['Yes; divide zero clicks by a tiny positive constant.','That returns a numerical value while leaving the relevance unidentified.'],
 ['Yes; all non-clicks prove the item is irrelevant.','If nobody examined it, a non-click does not distinguish relevant from irrelevant.']
], 'Worked example 1 — swapping two items', 'Add a zero-examination boundary case to the position-bias exercise.');

week(13, 'week-13-multi-agent-alphazero');
q('player-perspective','concept','In alternating zero-sum search, a child value is +1 from the next player’s perspective. What is that outcome worth to the current player?',[
 ['−1','The same outcome is the opponent’s win and the current player’s loss. Values must carry their player perspective through backup.'],
 ['+1','That silently treats both players as having the same objective, which is not the zero-sum convention.'],
 ['0','Changing perspective negates the value; it does not make a decisive result a draw.']
], 'Derivation: PUCT and value backup', 'Trace the sign change in MCTS._simulate on an immediate winning move.', 'project/alphazero.py');
q('puct-choice','calculate','PUCT uses −Qchild + P√Nparent/(1+Nchild). Parent visits = 16. A has (P,N,Q) = (0.6,3,−0.4), B = (0.4,0,0). Which score is larger?',[
 ['B: 1.6 versus A: 1.0','A = 0.4 + 0.6×4/4 = 1.0; B = 0 + 0.4×4/1 = 1.6. Exploration can overcome the current value estimate.'],
 ['A: its prior 0.6 exceeds B’s 0.4.','The prior is only one factor. B has no visits, so its exploration denominator is smaller.'],
 ['A: 2.8 versus B: 1.6','That omits A’s divisor 1 + Nchild = 4 in the exploration bonus.']
], 'Worked example 2 — exploration can override the current value estimate', 'Calculate these bonuses with exercise_1_puct and combine them with the correct player-perspective value.');
q('centralised-critic','transfer','A multi-agent system trains a centralised critic using global state but executes decentralised actors. What must be checked at deployment?',[
 ['Each actor’s action depends only on information actually available to that actor.','Extra critic inputs can help training without being legal actor inputs at execution. Audit the observation boundary.'],
 ['Every actor must receive the centralised critic’s global inputs online.','Centralised training does not require centralised execution when actors use local information.'],
 ['All team rewards must alternate sign between agents.','Sign alternation belongs to opposing zero-sum player perspectives; cooperative team rewards can be shared.']
], 'Worked example 3 — cooperation uses a different reward convention', 'Compare CentralisedCritic inputs with the actor inputs used during execution.', 'project/mappo.py');

week(14, 'week-14-offline-rl-1');
q('unsupported-maximum','concept','A fixed dataset contains action A with estimated Q = 2, but no examples of B; the model predicts Q(B) = 10. What is the danger in a naive greedy target?',[
 ['It can repeatedly exploit an unsupported high estimate for B.','Maximisation can amplify extrapolation error when no new environment interaction corrects it. A high prediction is not observed evidence.'],
 ['B must truly be better because 10 exceeds 2.','That confuses the learned estimate with a validated return.'],
 ['Offline data guarantees every action is evaluated equally.','Fixed logs can have narrow or missing action coverage.']
], 'Analogy: learning recipes from a fixed kitchen notebook', 'Compare optimistic unsupported values with the conservative penalty exercise.');
q('awr-frequency','calculate','At a fixed state, exact behaviour-weighted AWR uses pi(a) proportional to b(a)exp(beta A(a)). If both advantages are equal and b = (0.3,0.7), what is pi?',[
 ['(0.3,0.7)','Equal exponential factors cancel in normalisation, leaving the behaviour frequencies unchanged.'],
 ['(0.5,0.5)','That is softmax of equal values without the behaviour factor. It is a different objective.'],
 ['(0,1)','Equal advantages supply no relative preference beyond the existing behaviour distribution.']
], 'Worked example 2 — a missing behavior factor changes the policy', 'Compare normalised sample weights in exercise_2_awr_weights with policy probabilities after counting action frequencies.');
q('conservatism-evidence','transfer','A conservative offline learner lowers Q on unobserved actions. What does that repair establish by itself?',[
 ['It reduces one incentive to exploit extrapolation; it does not prove the learned policy is safe or optimal.','The penalty is a modelling/optimisation choice. Coverage, approximation and evaluation assumptions remain.'],
 ['It reveals the true return of every unobserved action.','A penalty cannot manufacture missing outcomes.'],
 ['It guarantees better performance than the logger on every dataset.','Finite samples, misspecification and optimisation can still produce poor policies; improvement needs evidence.']
], 'Code bridge and limits of this toy project', 'Ablate the CQL coefficient and report supported evaluation with uncertainty.');

week(15, 'week-15-decision-transformer-diffusion');
q('return-conditioning','concept','Every logged trajectory returns at most 4. You condition a Decision Transformer on desired return 10. What has been established?',[
 ['Only that the model was asked for an out-of-range condition, not that its policy will earn 10.','A desired-return token is an input, not a performance guarantee. Check coverage and actual rollouts.'],
 ['The generated trajectory must achieve 10 because the token is a constraint.','Supervised conditioning does not enforce that the environment delivers the requested return.'],
 ['The model has now collected evidence of return 10.','Changing an input does not add an observed trajectory to the dataset.']
], 'Worked example 1 — a negative reward raises remaining demand', 'Compare requested return with achieved return in exercise_5_conditional_generation.');
q('remaining-return','calculate','The requested undiscounted total return is 10. The first observed reward is −1. What remaining-return token preserves the original target?',[
 ['11','Remaining target = 10 − (−1) = 11. A loss increases what still needs to be earned.'],
 ['9','Adding the observed negative reward to the remaining target lowers the original objective instead of maintaining it.'],
 ['10','Keeping it unchanged ignores the −1 already accumulated.']
], 'Worked example 1 — a negative reward raises remaining demand', 'Compare logged returns-to-go from exercise_1_returns_to_go with deployment target updates.');
q('attention-leak','transfer','Tokens are [R₀,s₀,a₀,R₁,s₁,a₁]. A causal mask includes self-attention. Which hidden position can predict a₀ without directly seeing its answer token?',[
 ['Position 1, the s₀ token, attending only to positions 0–1.','It can use the desired return and current state while excluding the current action token. Earlier actions remain available at later state positions.'],
 ['Position 2, the a₀ token, because causal masking hides all answers.','Self-attention includes position 2 itself. A causal mask alone does not prevent answer leakage from a misplaced prediction head.'],
 ['Position 4, the s₁ token, because states never contain future information.','That position can attend to a₀ and later information. It leaks what the first action predictor should not see.']
], 'Worked example 2 — a six-token attention audit', 'Inspect DecisionTransformer.forward’s prediction positions as well as exercise_2_causal_mask.', 'project/model.py');

week(16, 'week-16-model-based');
q('model-validation','concept','A learned world model predicts its training trajectories accurately. What does that alone establish about a planner optimising inside it?',[
 ['Fit on those trajectories; planned actions may reach regions where the model is inaccurate.','Optimisation can exploit model errors outside observed coverage. Validate imagined predictions against fresh real transitions.'],
 ['Every imagined high-return plan will transfer to the environment.','That conclusion requires accuracy on the planner’s induced distribution, not merely the training distribution.'],
 ['A longer imagined horizon always improves real return.','Longer horizons can compound errors and increase exploitation of the model’s inaccuracies.']
], 'Code bridge and what the toy does not yet demonstrate', 'Inspect exercise_6_imagined_vs_real: it only summarises evaluation return. Add separate predicted-versus-observed rollouts to measure model error by depth.');
q('lambda-terminal','calculate','Use Gt = rt + gamma ct[(1−lambda)Vt+1 + lambda Gt+1]. With r=(1,2), V=(4,3,10), c=(1,0), gamma=0.9 and lambda=0.5, what is G₀?',[
 ['3.25','The terminal mask gives G₁ = 2. Then G₀ = 1 + 0.9[0.5×3 + 0.5×2] = 3.25.'],
 ['7.3','That propagates the terminal-state value: the incorrect G₁ = 11 then contaminates G₀. The terminal continuation must vanish.'],
 ['2.8','That is the lambda = 1 result: 1 + 0.9×2. At lambda 0.5, V₁ still contributes.']
], 'Worked example 2 — a terminal boundary stops propagation', 'Pass this fixture to exercise_2_lambda_return and vary only the continuation mask.');
q('kl-direction','transfer','Posterior q = Normal(1,1) and prior p = Normal(0,4), with the second argument denoting variance. May you swap KL(q||p) and KL(p||q) without changing the regulariser?',[
 ['No; their values and gradients generally differ.','KL is directed. Here q||p is about 0.443 nats, while p||q is about 1.307 nats.'],
 ['Yes; all distances between two Gaussians are symmetric.','KL is not a metric and need not be symmetric, even for simple Gaussians.'],
 ['Yes; both distributions have the same number of parameters.','Matching parameter counts says nothing about the likelihood ratio’s expectation under different sampling laws.']
], 'Worked example 1 — Gaussian disagreement and KL direction', 'Run exercise_1_kl_diag_gaussian in both directions, passing standard deviations 1 and 2.');

week(17, 'week-17-exploration-meta');
q('novelty-goal','concept','An RND agent visits a noisy but unrewarding room because its prediction error is high. What should you distinguish?',[
 ['Novelty under the predictor’s representation from success on the external task.','The intrinsic bonus changes the training objective. High novelty need not mean useful exploration or high task return.'],
 ['A high RND bonus proves the room has high external reward.','The bonus is prediction error, not a measurement of task utility.'],
 ['A frozen random target makes noisy observations predictable after one visit.','Stochastic observations and limited predictors can sustain prediction error. Freezing the target does not remove all uncertainty.']
], 'Notebook investigation: separate novelty, goal success and adaptation', 'Plot external return and intrinsic reward separately when running exercise_3_train_rnd.');
q('rnd-update','calculate','A fixed target is (1,−1), prediction p=(0,0), loss ||p−target||² and learning rate 0.25. What is the squared-error bonus after one gradient step on p?',[
 ['0.5','Gradient = (−2,2), so p′=(0.5,−0.5). Squared residuals sum to 0.25+0.25=0.5.'],
 ['1','That confuses Euclidean length with squared-error loss, or misses part of the gradient scaling.'],
 ['2','That is the initial bonus. The specified gradient step improves this simple parameterisation.']
], 'Worked example 1: a curiosity bonus decays on one fixed observation', 'Reproduce the direct-parameter calculation before inspecting exercise_2_intrinsic_decreases.');
q('maml-jacobian','transfer','For one exact MAML inner gradient step theta′ = theta − alpha∇Ltrain(theta), what factor connects the validation gradient back to theta?',[
 ['The inner-step Jacobian I − alpha Htrain, transposed when applying the chain rule.','Exact meta-gradients differentiate through adaptation. First-order MAML drops this Hessian-related factor as an approximation.'],
 ['Only the validation gradient at theta′; it is always the exact gradient at theta.','That drops the derivative of adaptation. It is a first-order approximation unless special conditions make the factor irrelevant.'],
 ['The reward baseline from REINFORCE.','Variance baselines and the chain rule through optimisation solve different problems.']
], 'Derivation: the gradient through one adaptation step', 'Inspect create_graph=True in _adapt, then reproduce the quadratic guide example.', 'project/maml.py');

week(18, 'week-18-rlhf-ppo-llm');
q('sample-kl-sign','concept','A sampled token has log p(token) − log q(token) < 0. Does that contradict nonnegativity of KL(p||q)?',[
 ['No; individual log-ratio samples can be negative while their expectation under p is nonnegative.','KL is an average over the sampling distribution. An underrepresented token contributes a negative sample.'],
 ['Yes; every token-level log ratio is a nonnegative KL divergence.','A selected token’s log ratio is a sample term, not the full categorical KL.'],
 ['No; KL(p||q) itself is allowed to be negative.','The expectation is nonnegative under the usual support conditions. The distinction is sample versus expectation.']
], 'Worked example 1: a negative sample and a positive KL', 'Compare exact categorical KL in exercise 2 with sampled token log-ratio penalties.');
q('rlhf-shaped','calculate','Sampled token log-ratios to the reference are (0.2,−0.1,0.3), beta=0.1 and a terminal score of 1 is added to the last token. What are the shaped rewards?',[
 ['(−0.02, +0.01, 0.97)','Each token receives −beta×log-ratio; only the final token also receives the terminal score.'],
 ['(0.98, 1.01, 0.97)','That adds the terminal sequence score to every token, changing the total reward.'],
 ['(+0.02, −0.01, 1.03)','That reverses the reference-penalty sign and rewards greater log-ratio rather than penalising it in expectation.']
], 'Worked example 2: three tokens, shaped rewards, GAE, clipping', 'Trace the selected-token probabilities and terminal-score placement in the PPO workbook.');
q('old-reference-policy','transfer','During PPO-based RLHF, which distribution belongs in the PPO probability ratio denominator?',[
 ['The policy that generated the rollout, held fixed for that batch.','The old rollout policy defines the importance ratio. The reference policy serves a distinct regularisation role.'],
 ['Always the original supervised reference policy.','That reference supplies a KL anchor. It need not be the policy that sampled the current rollout.'],
 ['The reward model’s preferred-token probability.','The reward model supplies scores or preferences, not the behaviour action probabilities needed for PPO.']
], 'Notebook investigation: trace one response through all four probability roles', 'Label policy, rollout-old policy, reference and reward model separately in project/ppo.py.', 'project/ppo.py');

week(19, 'week-19-dpo-zoo');
q('dpo-likelihood','concept','A DPO pairwise margin improves. Does the chosen response’s absolute probability necessarily increase?',[
 ['No; its probability can fall if the rejected response falls enough more.','The relative chosen-versus-rejected change can improve while probability mass moves to other responses. Track both quantities.'],
 ['Yes; DPO directly constrains every chosen probability to rise.','The pairwise objective does not impose that per-response monotonic constraint.'],
 ['No; DPO never uses chosen-response probabilities.','It does use their log probabilities, but relative to rejected and reference probabilities.']
], 'Worked example 1: a DPO loss and a likelihood-displacement trap', 'Reproduce both log-probability pairs that have the same DPO margin but different chosen probabilities.');
q('dpo-beta','calculate','Policy log probabilities are (−3,−5) for chosen/rejected; reference values are (−3.5,−3.5). With beta=0.1, what is the DPO logit?',[
 ['0.2','Relative logs are 0.5 and −1.5; their difference is 2. Apply beta once: 0.1×2 = 0.2.'],
 ['0.02','That multiplies by beta twice. Either scale the implicit rewards or the raw margin, not both.'],
 ['2','That is the unscaled reference-relative margin before multiplying by beta.']
], 'Derivation: DPO without counting beta twice', 'Compare the raw margin with the beta-scaled logit in dpo_loss.', 'project/losses.py');
q('length-normalisation','transfer','Two responses have log probabilities −3 over 10 tokens and −5 over 30 tokens. Which has the higher average token log probability?',[
 ['The 30-token response: −5/30 ≈ −0.167 exceeds −3/10 = −0.3.','Sequence probability and length-normalised token score rank these responses differently. State which score your objective uses.'],
 ['The 10-token response, because −3 exceeds −5.','That compares full-sequence log probabilities without dividing by length.'],
 ['They must tie because both are negative.','Negativity alone does not determine equality or ranking. Normalise each by its own token count.']
], 'Worked example 2: DPO, IPO, and SimPO measure different things', 'Compare DPO and SimPO losses on unequal lengths without treating loss magnitude as response quality.', 'project/losses.py');

week(20, 'week-20-grpo-rloo-rlvr');
q('rloo-independence','concept','Why does a leave-one-out reward baseline preserve the expected score-function gradient for i.i.d. completions from one prompt?',[
 ['The other completions’ rewards are independent of this completion’s score conditional on the prompt.','The baseline term then has zero expected product with this sample’s mean-zero score. Independence and grouping assumptions matter.'],
 ['Subtracting any group statistic always preserves the gradient.','Including the current reward in a simple group mean generally shrinks the expected gradient by (G−1)/G.'],
 ['RLOO must reduce gradient variance on every problem.','Unbiasedness does not guarantee variance reduction; the guide gives a counterexample.']
], 'Derivation: why excluding the current sample matters', 'Inspect rloo_advantages and enumerate two-completion groups.', 'project/train.py');
q('rloo-values','calculate','Four same-prompt rewards are (1,0,0,0). What are their leave-one-out advantages?',[
 ['(1, −1/3, −1/3, −1/3)','The successful sample’s peer mean is 0. Each failure’s three peers have mean 1/3.'],
 ['(0.75, −0.25, −0.25, −0.25)','Those subtract the inclusive group mean 0.25, not the leave-one-out mean.'],
 ['(√3, −1/√3, −1/√3, −1/√3)','Those are population-standardised centred advantages (ignoring epsilon), not RLOO.']
], 'Worked example 1: four completions, three kinds of coefficient', 'Compare rloo_advantages and grpo_advantages on the same reward group.', 'project/train.py');
q('verifier-validity','transfer','An exact-match verifier gives all generated answers reward 1. What does a zero centred group advantage tell you?',[
 ['There is no within-group reward contrast; it does not establish that the verifier measures the intended quality.','Equal rewards cancel in the centred numerator. Audit parsing and correctness separately; regularisation can still update the policy.'],
 ['The policy is optimal for all prompts.','One group’s identical verifier outputs cannot establish global task performance or verifier validity.'],
 ['Dividing by a tiny epsilon creates a strong learning signal.','A zero numerator remains zero. Epsilon stabilises division, not reward information.']
], 'Notebook investigation: make the grouping axis and verifier visible', 'Test extract_final_number and exact_match_reward on malformed and adversarial answers.', 'project/env.py');

week(21, 'week-21-reward-hacking-oversight');
q('proxy-quality','concept','Why can optimising a noisy proxy select outcomes whose measured score substantially exceeds their expected true quality?',[
 ['Selection can favour positive measurement noise as well as genuine quality.','Under the guide’s Gaussian independent-noise model, expected quality regresses toward the mean. Other noise models need their own reasoning.'],
 ['Every high proxy score proves deliberate cheating.','Regressional selection effects can occur without an agent intentionally exploiting a loophole.'],
 ['A proxy score and true quality always share the same scale and units.','They may not. A raw score gap can be meaningless without a scale definition.']
], 'Derivation: regressional Goodhart under stated assumptions', 'Separate proxy score from an independently defined quality measure in the hack lab.', 'project/hack_lab.py');
q('goodhart-shrinkage','calculate','Q and noise E are independent zero-mean Gaussians with variances 1 and 3. P = Q+E. What is E[Q | P=4]?',[
 ['1','The coefficient is Var(Q)/Var(P) = 1/(1+3) = 1/4, so expected quality is 1.'],
 ['4','That treats all selected proxy score as quality and ignores measurement noise.'],
 ['3','That is the expected noise contribution, (3/4)×4, rather than expected quality.']
], 'Worked example 1: a high score does not mean equally high quality', 'Simulate independent Gaussian quality/noise and compare conditional averages near P=4 with the analytic result.');
q('partial-proxy-fix','transfer','A proxy rewards text length and keyword repetition. A length penalty removes the length incentive, but repeated keywords still increase score. What is the next useful test?',[
 ['Construct a keyword-only counterexample and evaluate it with an independent quality criterion.','A repair to one proxy component need not repair another. Test the remaining incentive directly.'],
 ['Declare reward hacking solved because the average output became shorter.','Shorter outputs may still game keywords. Length is only one diagnostic.'],
 ['Treat a lower proxy score as automatic evidence of higher real quality.','The proxy is the disputed measure. Its decrease cannot independently validate quality improvement.']
], 'Worked example 2: a length penalty can leave keyword gaming intact', 'Compare “the cat sat” and “goodgoodgood” before and after a length penalty in proxy_reward.', 'project/hack_lab.py');

week(22, 'week-22-cmdp-cpo-lagrangian');
q('constraint-expectation','concept','A policy satisfies an expected episode-cost budget. What does that imply about individual episodes?',[
 ['Some episodes can still exceed the budget; an expectation constraint is not a per-episode guarantee.','The distribution can mix low- and high-cost outcomes while meeting the mean. Choose the risk definition that matches the task.'],
 ['Every episode must remain below the budget.','That is a stronger requirement than bounding expected cost.'],
 ['It must always choose the lowest-cost deterministic action.','Randomisation can satisfy the mean budget while achieving greater reward.']
], 'Worked example 1: the optimal constrained policy may mix actions', 'Plot episode-cost distribution as well as its mean in the constrained-policy exercise.');
q('multiplier-sign','calculate','Lambda = 0.5, multiplier rate = 0.2, observed cost = 1.5 and budget = 1. Using lambda′ = max(0, lambda + rate×(cost−budget)), what is lambda′?',[
 ['0.6','The violation is +0.5, so the cost price rises by 0.2×0.5 = 0.1.'],
 ['0.4','That lowers the price after a violation, reversing the intended response.'],
 ['0.8','That multiplies by total cost rather than excess above the budget.']
], 'Worked example 2: follow one multiplier and advantage calculation', 'Run exercise_1 with overspending, underspending and projection-to-zero cases.');
q('cost-units','transfer','A safety budget is defined as total episode cost, but the multiplier is updated from mean cost per timestep. What needs fixing first?',[
 ['Align the estimated cost and the budget’s units, horizon and discount convention.','Comparing different quantities can reward an infeasible policy or penalise a feasible one even when the update formula is correct.'],
 ['Increase the multiplier learning rate until the numbers look closer.','A faster update cannot fix mismatched definitions.'],
 ['Normalise reward and assume this also fixes the cost budget.','Reward scaling and cost-budget semantics are separate.']
], 'Bridge to the actual project', 'Inspect _collect’s episode-cost accumulation and the statistic used in the multiplier update.', 'project/train.py');

week(23, 'week-23-safe-exploration-shielding-robust');
q('one-step-trap','concept','A shield rejects immediately dangerous successors but allows entering a currently safe state from which every next action is dangerous. What is missing?',[
 ['A viability/invariant-set condition ensuring a safe continuation remains available.','Immediate labels are insufficient: remove states that cannot stay safe, then allow actions that stay inside the resulting set.'],
 ['A larger reward bonus for entering that state.','Reward shaping does not create a safe action where none exists.'],
 ['Nothing; a safe next state guarantees all later states are safe.','That inference fails in the described trap one step later.']
], 'Worked example 1: the one-step-safe trap', 'Construct start → trap → danger and compare immediate filtering with backward safe-set pruning.', 'project/shield.py');
q('barrier-direction','calculate','Choose the closest action to a₀=1 subject to −2a ≥ −1, with no other limits. What is the projected action?',[
 ['0.5','Dividing by −2 reverses the inequality: a ≤ 0.5. The nearest feasible point to 1 is the boundary 0.5.'],
 ['1','The proposed action violates −2 ≥ −1, which is false.'],
 ['−0.5','That is feasible but unnecessarily far from the proposed action; the minimum-change projection is 0.5.']
], 'Worked example 2: the direction of the inequality matters', 'Run exercise_7 with positive, negative and zero barrier coefficients.');
q('shield-model-limit','transfer','A shield is proved safe for a known deterministic transition model. Real dynamics sometimes slip to an unmodelled unsafe state. Does the proof cover those slips?',[
 ['No; the guarantee depends on the transition model/support assumptions used in the proof.','For a worst-case stochastic guarantee, every possible successor must remain safe. Unknown transitions break that premise.'],
 ['Yes; a proof about one model covers every environment.','The inductive guarantee uses the model’s successors, so changing them changes the premise.'],
 ['Yes; low-probability failures count as deterministic safety.','Rare risk and impossibility of violation are different claims.']
], 'Derivation: from immediate safety to an invariant set', 'Add a slipped transition to the shield fixture and state which guarantee still survives.', 'project/shield.py');

week(24, 'week-24-capstone-safe-matching');
q('one-step-iql','concept','In the capstone’s one-step contextual bandit, why is the Q-regression target just the observed reward?',[
 ['The episode ends after that decision, so there is no continuation return.','This is the terminal special case of a Bellman target. Sequential deployments would require a different model.'],
 ['Offline learning always ignores future rewards.','Offline RL can be sequential. This target is immediate because of the one-step task definition.'],
 ['Expectile regression removes the need to model any rewards.','Expectiles fit a value summary from Q estimates; they do not replace the reward target.']
], 'Derivation: IQL in a one-step bandit', 'Inspect the reward regression and expectile objectives separately in offline_iql.py.', 'project/offline_iql.py');
q('capstone-snips','calculate','Four records have weights (1.5,2,0.5,1/3) and rewards (1,0,1,0). What is SNIPS?',[
 ['6/13 ≈ 0.4615','Weighted reward sum is 2; weights sum to 13/3. Their ratio is 2/(13/3) = 6/13.'],
 ['0.5','That is IPS here, dividing the weighted reward sum by four records rather than the sum of weights.'],
 ['2','That is the weighted reward sum before normalisation.']
], 'Worked example 2: SNIPS with the propensity of the action actually taken', 'Pass selected-action target probabilities to snips_ope, not the probability of a preferred alternative.', 'project/audit.py');
q('audit-policy-owner','transfer','An audit combines offline estimates, online rollout rewards and fairness summaries. What makes its comparisons interpretable?',[
 ['Label each number with its policy, population, outcome, denominator and evaluation split.','These quantities can describe different decisions and samples. Their names alone do not establish a common estimand.'],
 ['Put them on one chart with the same colour.','Visual proximity does not align policy ownership, units or populations.'],
 ['Call every result “policy performance” and average them.','Averaging different estimands hides the distinctions needed to assess evidence.']
], 'Notebook investigation: build an audit in which every number has an owner', 'Build a metric-to-policy table alongside the audit output.', 'project/audit.py');

week(25, 'week-25-synthetic-ats-rl-audit');
q('shortlist-support','concept','A logger explores randomly within a heuristic shortlist. Does epsilon exploration provide support for candidates excluded from that shortlist?',[
 ['No; its randomisation only covers eligible actions inside the shortlist.','Selection before randomisation defines a separate support boundary. Record the eligible set and analyse excluded candidates separately.'],
 ['Yes; any positive epsilon covers every candidate in the population.','Epsilon is distributed only over the action set actually offered to the logger.'],
 ['Yes; a later reward model can retrospectively add logged support.','Model predictions are not logged randomised outcomes for excluded actions.']
], 'Bridge to the actual project: read these boundaries first', 'Inspect collect_logs and the shortlist construction before interpreting OPE.', 'project/train.py');
q('dr-contributions','calculate','Behaviour=(0.5,0.5), target=(0.75,0.25), logged actions=(A,B,A), rewards=(1,0,0), and fixed q-hat=(0.6,0.2). What is the DR estimate?',[
 ['11/30 ≈ 0.3667','The target model value is 0.5. Residual-corrected terms are 1.1, 0.4 and −0.4; their mean is 1.1/3.'],
 ['0.5','That is the direct model value and, coincidentally, IPS on these rows; it omits the nonzero mean residual correction.'],
 ['3/7 ≈ 0.4286','That is SNIPS: weighted rewards 1.5 divided by weights 3.5. DR uses another construction.']
], 'Worked example 1: calculate three estimators and an ESS', 'Reproduce the three DR row contributions before averaging them in the OPE helper.', 'project/ope.py');
q('shadow-causality','transfer','A shadow scorer agrees with every displayed candidate choice but never changes which candidate is shown. What has been measured?',[
 ['Agreement with the active policy, not the shadow scorer’s causal impact on outcomes.','The active policy generated the outcomes. Impact needs a supported evaluation design or an intervention with appropriate assumptions.'],
 ['The causal improvement the shadow scorer would produce if deployed.','Identical shadow scores do not create counterfactual outcome evidence.'],
 ['Proof that every eligible candidate had equal selection probability.','Agreement tells you nothing by itself about propensities or exposure denominators.']
], 'Notebook investigation: follow the evaluated policy, not the result-key name', 'Label behaviour choices, fitted scorer and fixed OPE target separately in the training output.', 'project/train.py');

week(26, 'week-26-causal-discovery-biological-data');
q('prediction-causation','concept','A model improves held-out prediction of a biological outcome from treatment and markers. What does that alone establish?',[
 ['Predictive performance on that test distribution, not an identified intervention effect.','Causal claims need a defined intervention, target population and identification assumptions. Prediction accuracy does not supply them.'],
 ['The treatment causes the predicted outcome change.','Confounding or proxies can support prediction without identifying what happens when treatment is set.'],
 ['The learned graph has the unique true causal direction.','Different causal models can share the same observational distribution.']
], 'Four questions with four different answers', 'Compare naive and adjusted effects before interpreting predictive metrics in the workbook.');
q('standardisation','calculate','Mild patients have treated/untreated mean outcomes (3,2); severe patients have (1,0). The target population is 50% mild and 50% severe. What is the standardised effect?',[
 ['1','Use the same target weights for both arms: 0.5(3−2)+0.5(1−0)=1. A causal interpretation still needs consistency, exchangeability and positivity.'],
 ['−0.2','That is the pooled association in the guide’s unequal-treatment-mixture example, not the standardised contrast.'],
 ['2','That adds stratum effects without weighting them by their target-population probabilities.']
], 'Numerical example 1: confounding reverses a decision', 'Run exercise_2_adjusted_ate on the severity fixture and test a missing treatment cell.');
q('patient-leakage','transfer','A biological dataset contains multiple rows per patient. You randomly split rows and fit imputation on all rows. What needs changing before trusting test performance?',[
 ['Split by patient and fit preprocessing only on the training partition.','Repeated patients and full-data preprocessing can leak information across the evaluation boundary. Keep a held-out environment separate when testing shift.'],
 ['Increase model size so it handles the duplicate patients.','Capacity does not repair leakage; it may make memorisation easier.'],
 ['Keep the split, but hide patient identifiers from the final report.','Removing a displayed identifier does not remove shared biological information or preprocessing leakage.']
], 'Code bridge: one bounded experiment', 'Use exercise_4_patient_split, then fit exercise_5_impute_fit on training features only.');

// Verified source-line and zero-based source-notebook indices. Published cell
// labels/anchors are one-based, so conversion happens below. Tests check both.
const HANDOFF_TARGETS = {
  "w01-reward-regret": {
    "line": 39,
    "symbol": "regret",
    "cell": 7,
    "needle": "Exercise 2 (★) — $\\epsilon$-greedy cumulative regret"
  },
  "w01-arm-count": {
    "line": 24,
    "symbol": "exercise_1_incremental_mean",
    "cell": 5,
    "needle": "exercise_1_incremental_mean"
  },
  "w02-evaluation-improvement": {
    "line": 12,
    "symbol": "exercise_1_bellman_operator",
    "cell": 5,
    "needle": "exercise_1_bellman_operator"
  },
  "w02-continuing-loop": {
    "cell": 23,
    "needle": "Exercise 6 (★★) — Policy evaluation as a linear system"
  },
  "w02-stable-policy": {
    "line": 80,
    "symbol": "exercise_7_gpi",
    "cell": 29,
    "needle": "exercise_7_gpi"
  },
  "w03-first-visit-order": {
    "line": 15,
    "symbol": "exercise_1_first_visit_returns",
    "cell": 5,
    "needle": "exercise_1_first_visit_returns"
  },
  "w03-first-return": {
    "line": 15,
    "symbol": "exercise_1_first_visit_returns",
    "cell": 5,
    "needle": "exercise_1_first_visit_returns"
  },
  "w03-trajectory-support": {
    "line": 108,
    "symbol": "exercise_4_ordinary_is",
    "cell": 17,
    "needle": "exercise_4_ordinary_is"
  },
  "w04-on-off-policy": {
    "cell": 11,
    "needle": "3. SARSA single-step update (one star)"
  },
  "w04-terminal-target": {
    "cell": 3,
    "needle": "1. TD(0) prediction (one star)"
  },
  "w05-detached-target": {
    "line": 39,
    "symbol": "DQNAgent",
    "cell": 29,
    "needle": "DQNAgent"
  },
  "w05-linear-gradient": {
    "line": 109,
    "symbol": "exercise_8_smoke_dqn",
    "cell": 27,
    "needle": "exercise_8_smoke_dqn"
  },
  "w05-batch-shape": {
    "line": 12,
    "symbol": "exercise_1_gather_q",
    "cell": 6,
    "needle": "exercise_1_gather_q"
  },
  "w06-double-selection": {
    "line": 128,
    "symbol": "RainbowAgent",
    "cell": 3,
    "needle": "1. Double DQN target (two stars)"
  },
  "w06-per-correction": {
    "cell": 9,
    "needle": "3. PER importance-sampling weight (two stars)"
  },
  "w06-distribution-uncertainty": {
    "cell": 19,
    "needle": "6. C51 categorical projection (three stars)"
  },
  "w07-baseline-mean": {
    "cell": 6,
    "needle": "Exercise 2 (★) — Score function has zero mean"
  },
  "w07-bernoulli-score": {
    "cell": 12,
    "needle": "Exercise 4 (★★) — REINFORCE on a two-armed bandit"
  },
  "w07-baseline-variance": {
    "cell": 14,
    "needle": "Exercise 5 (★★) — Variance reduction with $V_\\phi$"
  },
  "w08-lambda-one": {
    "line": 32,
    "symbol": "compute_gae",
    "cell": 6,
    "needle": "compute_gae"
  },
  "w08-gae-residuals": {
    "line": 32,
    "symbol": "compute_gae",
    "cell": 6,
    "needle": "compute_gae"
  },
  "w08-trpo-guarantee": {
    "cell": 17,
    "needle": "Exercise 8 (★★★, *meaningful objective*) — Trust-region weekly retrain"
  },
  "w09-negative-advantage": {
    "line": 10,
    "symbol": "ppo_clip_loss",
    "cell": 5,
    "needle": "ppo_clip_loss"
  },
  "w09-ppo-loss-sign": {
    "line": 10,
    "symbol": "ppo_clip_loss",
    "cell": 5,
    "needle": "ppo_clip_loss"
  },
  "w09-clip-not-bound": {
    "line": 64,
    "symbol": "PPOAgent",
    "cell": 14,
    "needle": "Exercise 6 (★★) — Track the approximate KL"
  },
  "w10-temperature-scale": {
    "cell": 6,
    "needle": "Exercise 2 (★) — Derive the soft-optimal policy"
  },
  "w10-soft-terminal": {
    "line": 54,
    "symbol": "soft_bellman_target",
    "cell": 9,
    "needle": "soft_bellman_target"
  },
  "w10-action-rescale": {
    "line": 13,
    "symbol": "tanh_gaussian_log_prob",
    "cell": 5,
    "needle": "tanh_gaussian_log_prob"
  },
  "w11-ope-overlap": {
    "line": 58,
    "symbol": "snips",
    "cell": 16,
    "needle": "Exercise 5 — IPS and SNIPS for a uniform target (★★)"
  },
  "w11-ips-range": {
    "line": 50,
    "symbol": "ips",
    "cell": 16,
    "needle": "Exercise 5 — IPS and SNIPS for a uniform target (★★)"
  },
  "w11-dr-limits": {
    "line": 79,
    "symbol": "doubly_robust",
    "cell": 19,
    "needle": "Exercise 6 — Doubly Robust against softmax target (★★)"
  },
  "w12-click-relevance": {
    "line": 37,
    "symbol": "ips_ltr_value",
    "cell": 13,
    "needle": "Exercise 4 — IPS-LTR vs.\\ naive click count (★★)"
  },
  "w12-selection-denominator": {
    "cell": 19,
    "needle": "Exercise 6 — Demographic-parity gap (★★)"
  },
  "w12-zero-examination": {
    "cell": 7,
    "needle": "Exercise 2 — Position-bias model (★)"
  },
  "w13-player-perspective": {
    "line": 89,
    "symbol": "MCTS",
    "cell": 7,
    "needle": "Exercise 2 — Terminal value (★)"
  },
  "w13-puct-choice": {
    "line": 16,
    "symbol": "exercise_1_puct",
    "cell": 6,
    "needle": "exercise_1_puct"
  },
  "w13-centralised-critic": {
    "line": 44,
    "symbol": "CentralisedCritic",
    "cell": 22,
    "needle": "Exercise 7 — MAPPO on the coordination game (★★)"
  },
  "w14-unsupported-maximum": {
    "cell": 10,
    "needle": "Exercise 3 (★★) — CQL penalty"
  },
  "w14-awr-frequency": {
    "line": 20,
    "symbol": "exercise_2_awr_weights",
    "cell": 9,
    "needle": "exercise_2_awr_weights"
  },
  "w14-conservatism-evidence": {
    "cell": 19,
    "needle": "Exercise 6 (★★) — CQL ablation over $\\alpha$"
  },
  "w15-return-conditioning": {
    "line": 32,
    "symbol": "exercise_5_conditional_generation",
    "cell": 18,
    "needle": "exercise_5_conditional_generation"
  },
  "w15-remaining-return": {
    "line": 12,
    "symbol": "exercise_1_returns_to_go",
    "cell": 6,
    "needle": "exercise_1_returns_to_go"
  },
  "w15-attention-leak": {
    "line": 18,
    "symbol": "DecisionTransformer",
    "cell": 7,
    "needle": "Ex 2 (★) Causal mask"
  },
  "w16-model-validation": {
    "line": 46,
    "symbol": "exercise_6_imagined_vs_real",
    "cell": 21,
    "needle": "exercise_6_imagined_vs_real"
  },
  "w16-lambda-terminal": {
    "line": 16,
    "symbol": "exercise_2_lambda_return",
    "cell": 9,
    "needle": "exercise_2_lambda_return"
  },
  "w16-kl-direction": {
    "line": 11,
    "symbol": "exercise_1_kl_diag_gaussian",
    "cell": 6,
    "needle": "exercise_1_kl_diag_gaussian"
  },
  "w17-novelty-goal": {
    "line": 32,
    "symbol": "exercise_3_train_rnd",
    "cell": 12,
    "needle": "exercise_3_train_rnd"
  },
  "w17-rnd-update": {
    "line": 18,
    "symbol": "exercise_2_intrinsic_decreases",
    "cell": 9,
    "needle": "exercise_2_intrinsic_decreases"
  },
  "w17-maml-jacobian": {
    "line": 38,
    "symbol": "_adapt",
    "cell": 19,
    "needle": "Ex 6 (★★) Inner adaptation improves expected reward"
  },
  "w18-sample-kl-sign": {
    "cell": 8,
    "needle": "Exercise 2 (star) -- token-level KL between two categorical distributions"
  },
  "w18-rlhf-shaped": {
    "cell": 16,
    "needle": "Exercise 4 (star star) -- GAE"
  },
  "w18-old-reference-policy": {
    "cell": 12,
    "needle": "Exercise 3 (star star) -- PPO clipped surrogate"
  },
  "w19-dpo-likelihood": {
    "cell": 8,
    "needle": "Exercise 2 (star star) -- DPO loss from scratch"
  },
  "w19-dpo-beta": {
    "line": 23,
    "symbol": "dpo_loss",
    "cell": 8,
    "needle": "Exercise 2 (star star) -- DPO loss from scratch"
  },
  "w19-length-normalisation": {
    "cell": 16,
    "needle": "Exercise 4 (star star) -- SimPO loss"
  },
  "w20-rloo-independence": {
    "line": 51,
    "symbol": "rloo_advantages",
    "cell": 8,
    "needle": "Exercise 2 ($\\bigstar\\bigstar$). RLOO advantage from a vector of group rewards."
  },
  "w20-rloo-values": {
    "line": 32,
    "symbol": "grpo_advantages",
    "cell": 8,
    "needle": "Exercise 2 ($\\bigstar\\bigstar$). RLOO advantage from a vector of group rewards."
  },
  "w20-verifier-validity": {
    "line": 21,
    "symbol": "extract_final_number",
    "cell": 4,
    "needle": "Exercise 1 ($\\bigstar$). Extract the final integer from a completion."
  },
  "w21-proxy-quality": {
    "cell": 8,
    "needle": "Exercise 2 ($\\bigstar\\bigstar$). Proxy vs true quality disagreement."
  },
  "w21-goodhart-shrinkage": {
    "cell": 4,
    "needle": "Exercise 1 ($\\bigstar$). Classify a Goodhart scenario."
  },
  "w21-partial-proxy-fix": {
    "line": 100,
    "symbol": "proxy_reward",
    "cell": 14,
    "needle": "Exercise 4 ($\\bigstar\\bigstar$). Length penalty."
  },
  "w22-constraint-expectation": {
    "cell": 16,
    "needle": "Exercise 5 ★★ — Empirical CVaR"
  },
  "w22-multiplier-sign": {
    "line": 26,
    "symbol": "exercise_1",
    "cell": 5,
    "needle": "exercise_1"
  },
  "w22-cost-units": {
    "line": 25,
    "symbol": "_collect",
    "cell": 19,
    "needle": "Exercise 6 ★★ — Meaningful objective: train PPO-Lagrangian and audit"
  },
  "w23-one-step-trap": {
    "cell": 10,
    "needle": "Exercise 3 ★★ — Pre-shield filtering"
  },
  "w23-barrier-direction": {
    "line": 91,
    "symbol": "exercise_7",
    "cell": 23,
    "needle": "exercise_7"
  },
  "w23-shield-model-limit": {
    "cell": 16,
    "needle": "Exercise 5 ★★ — Check a finite rollout against the stated safety property"
  },
  "w24-one-step-iql": {
    "cell": 10,
    "needle": "Exercise 3 ★★ — IQL pretrain"
  },
  "w24-capstone-snips": {
    "line": 53,
    "symbol": "snips_ope",
    "cell": 13,
    "needle": "Exercise 4 ★★ — SNIPS off-policy evaluation"
  },
  "w24-audit-policy-owner": {
    "cell": 19,
    "needle": "Exercise 6 ★★★ — Meaningful objective: end-to-end audit"
  },
  "w25-shortlist-support": {
    "line": 56,
    "symbol": "collect_logs",
    "cell": 19,
    "needle": "E6 — Propensity-logging patch sketch (★★)"
  },
  "w25-dr-contributions": {
    "cell": 13,
    "needle": "E4 — SNIPS confidence interval (★★)"
  },
  "w25-shadow-causality": {
    "cell": 31,
    "needle": "E10 — Phase-1 shadow-mode reranker (★★★, meaningful objective)"
  },
  "w26-prediction-causation": {
    "cell": 3,
    "needle": "Exercise 1 ★: distinguish observed association from an effect"
  },
  "w26-standardisation": {
    "line": 34,
    "symbol": "exercise_2_adjusted_ate",
    "cell": 7,
    "needle": "exercise_2_adjusted_ate"
  },
  "w26-patient-leakage": {
    "line": 68,
    "symbol": "exercise_4_patient_split",
    "cell": 13,
    "needle": "exercise_4_patient_split"
  }
};
for (const row of rows) {
  const target = HANDOFF_TARGETS[row.id];
  if (!target) continue;
  if (target.line) { row.handoff.source += `#L${target.line}`; row.handoff.sourceLine = target.line; row.handoff.sourceSymbol = target.symbol; }
  if (Number.isInteger(target.cell)) { const cell = target.cell + 1; row.handoff.route += `&section=cell-${cell}`; row.handoff.cell = cell; row.handoff.cellNeedle = target.needle; row.handoff.label = `Open week ${row.week} workbook at the related exercise`; }
}

export const QUESTIONS = Object.freeze(rows.map(row => Object.freeze({...row,
  choices:Object.freeze(row.choices), feedback:Object.freeze(row.feedback),
  repair:Object.freeze(row.repair), handoff:Object.freeze(row.handoff)})));
export const QUESTION_BY_ID = Object.freeze(Object.assign(Object.create(null),Object.fromEntries(QUESTIONS.map(question => [question.id,question]))));
export function questionsForWeek(value) {
  const number = String(Number(value)).padStart(2,'0');
  return QUESTIONS.filter(question => question.week === number);
}
