# Learn RL by changing one thing at a time

Start by choosing an action and observing what happens. Add future rewards only
after that feels natural. Add a neural network only after you can calculate a
table update yourself. The [learning path](LEARNING_PATH.md) connects these small
experiments to every week; this walkthrough gives you the activities and coding
tasks. You do not need to complete all of them in your first session.

The [browser learning lab](../interactive/index.html) runs locally with graphs,
step controls, and short prediction checks. Open that file directly in a modern
browser, or serve it from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Visit `http://localhost:8000/interactive/` in your browser. Serve the repository
root so links to weekly guides and project files work. Stop the server with Ctrl+C when
finished. The lab's numbers are calculated by the small teaching simulations;
they are not benchmark results from large RL models. The Python projects below
use no extra packages and can run alongside the lab.

## Your first 20 minutes: three routes, one average

This is the week 1 beginner route. Leave future-value notation, posterior
distributions, UCB proofs, neural networks, and logged-policy evaluation for later.

1. Open [**1 · Try a route**](../interactive/index.html#bandit). Each route is an action. Choosing it reveals
   one delivery outcome; it does not reveal the other routes' outcomes.
2. Make a prediction before each of five choices: “I expect route ___ to be good
   because ___.” A success is one observation, not proof of a good average.
3. On paper, average observations `[1, 0]`. The answer is 0.5. Now add a success:
   `[1, 0, 1]` averages to 2/3. Count that route's observations, not all choices.
4. Implement only `mean_update` in [exercises.py](../learning_projects/exercises.py).
   Run this focused check from the repository root:

   ```bash
   python -m learning_projects.check_exercises --exercise 1
   ```

5. Explain why a route with one success remains uncertain. Tomorrow, repeat the
   `[1, 0, 1]` calculation without looking at your code.

An unfinished exercise prints `PENDING` and exits with status 1. That is expected
before editing; an incorrect implementation prints `FAIL` with an input and its
expected answer. The checker never replaces your function with a supplied answer.
The four later exercises can stay unfinished while you learn week 1.

<details>
<summary>Exercise 1 hints — read one, then attempt the task again</summary>

<details>
<summary>Hint 1</summary>

the old mean summarizes `new_count - 1` observations of this arm.

</details>

<details>
<summary>Hint 2</summary>

recover the old sum by multiplying the old mean by that old count;
add the new reward and divide by `new_count`.

</details>

<details>
<summary>Hint 3</summary>

rearrange this into `old_mean + (reward - old_mean) / new_count`.
For old mean 1, reward 0, and new count 2, it becomes `1 + (0 - 1) / 2 = 0.5`.

</details>

**Transfer:** a reward of -3 after an old mean 1 with new count 2 should give -1.
An algorithm that always increases its estimate is using the error incorrectly.

</details>

When this works, read the early sections of
[week 1's teaching guide](../week-01-mdps-bandits/teaching_guide.md).
The posterior/UCB sections are stretch work; contextual bandits and OPE return
with full motivation in week 11. You can progress to week 2 without implementing
every optional first-week component.

## Weeks 2–3: a delivery now or a delivery later?

Keep the story tiny. **Stop** pays 2 immediately and ends the episode.
**Continue** pays 0, moves to the finish state, and then pays 4.
With discount 0.9, continuing returns `0 + 0.9 × 4 = 3.6`.
If you continue with probability 0.25, the policy's expected return is
`0.75 × 2 + 0.25 × 3.6 = 2.4`. An individual episode still returns either 2
or 3.6 in this deterministic world; 2.4 is an average across the policy's choices.

```bash
python -m learning_projects.two_step
python -m learning_projects.two_step --gamma 0.4
python -m learning_projects.two_step --gamma 0.6
```

**Predict first:** at which discount do stop and continue tie? Use the browser
lab's [**2 · Send value backward**](../interactive/index.html#bellman) stage to change discount and watch future reward affect the
earlier choice. The tie is at 0.5 because `4 × 0.5 = 2`.

Now implement `discounted_return` and check it:

```bash
python -m learning_projects.check_exercises --exercise 2
```

<details>
<summary>Exercise 2 hint ladder — when does the discount start?</summary>

<details>
<summary>Hint 1</summary>

the first reward is paid after the current action. It has weight 1.

</details>

<details>
<summary>Hint 2</summary>

the second reward has weight gamma; the third has weight gamma².

</details>

<details>
<summary>Hint 3</summary>

sum `gamma ** step * reward` over `enumerate(rewards)`.
For `[2, 1, 1]` and gamma 0.5, the return is `2 + 0.5 + 0.25 = 2.75`.

</details>

**Transfer:** an empty reward sequence returns 0; gamma 0 keeps just the first
reward; gamma 1 sums all rewards in this finite episode.

</details>

In [**3 · Learn before the ending**](../interactive/index.html#targets), keep the actual continued trajectory `[0, 4]`, an old
start-state estimate 1, a next-state estimate 3, and learning rate 0.5.
Both methods reuse week 1's correction pattern: `old + step_size × (target - old)`.
Here step size 0.5 moves halfway toward the target; the new ingredient is how
each method obtains that target.

```bash
python -m learning_projects.two_step --stage targets
```

| Question | Monte Carlo | TD(0) |
|---|---|---|
| What evidence is used? | Wait for the complete rewards `[0, 4]` | Use reward 0 and the next estimate 3 |
| Target | `0 + 0.9 × 4 = 3.6` | `0 + 0.9 × 3 = 2.7` |
| New estimate | `1 + 0.5 × (3.6 - 1) = 2.3` | `1 + 0.5 × (2.7 - 1) = 1.85` |

TD's smaller number here comes from an underestimated future, not a different
observed first reward. Set the next estimate to 4: the two targets agree in this
deterministic example. Read [week 3](../week-03-monte-carlo/teaching_guide.md)
for what changes when complete returns are random.

At week 7, use `python -m learning_projects.two_step --stage policy` to add
advantage: an action's value minus the current policy's average. Earlier stages
keep that extra concept out of the printed walkthrough.

## Week 4: play the courier, then teach it one update

The game adds positions and four actions to the same delivery idea. Coordinates
are `(column, row)`, starting at `(0, 0)` at the top left. `G` is the destination;
`!` is a hazard. Entering either ends the episode. Ordinary movement, including
bumping a boundary, pays -0.1; the goal pays 4; a hazard pays -3 and records cost 1.

```text
S . . .
. . . .
. ! ! .
. . . G
```

```bash
python -m learning_projects.courier --play
```

Use `w/a/s/d`, then Enter, for up/left/down/right. Use `q` to quit.
A safe six-move route is right, right, right, down, down, down. Its undiscounted
reward is five ordinary moves at -0.1 and a goal reward 4: **3.5**.
Try a route entering a hazard and explain why the episode stops.

In [**4 · Teach a tiny game**](../interactive/index.html#courier), inspect a cell's action
values and train a few episodes. Learning builds values from actual transitions;
the arrows are the greedy choices of the current table.

Before training Python, calculate this update: old `Q(s,a)=2`, observed reward
-1, gamma 0.9, next action values `[4, -2]`, learning rate 0.5.
Q-learning's target is `-1 + 0.9 × max(4, -2) = 2.6`; the updated value is 2.3.
Implement `q_target`, then run your own target in the game:

```bash
python -m learning_projects.check_exercises --exercise 3
python -m learning_projects.courier --learner --episodes 1200 --seed 7
```

<details>
<summary>Exercise 3 hint ladder — when must future value disappear?</summary>

<details>
<summary>Hint 1</summary>

ask whether the environment truly ended before reading the next values.

</details>

<details>
<summary>Hint 2</summary>

a true terminal target is just the observed reward. A nonterminal
target adds gamma times the best available next-action value.

</details>

<details>
<summary>Hint 3</summary>

return the reward when `terminated` is true; otherwise return
`reward + gamma * max(next_values)`.

</details>

**Transfer:** a terminal reward 5 with a stored future estimate 100 still has
target 5. An external collector cutoff with reward 1 and future estimate 10 has
target `1 + 0.9 × 10 = 10`. Stopping collection does not change the task.

</details>

The reference demo is separate:

```bash
python -m learning_projects.courier --episodes 1200 --seed 1
python -m learning_projects.courier --episodes 1200 --seed 7
python -m learning_projects.courier --episodes 1200 --seed 19
```

The CLI reports exploration hazard entries, fresh greedy evaluation, collector
cutoffs, and the first greedy path. On this small deterministic map, the supplied
reference learns a safe six-move route for those seeds. That is a checked toy
outcome, not a theorem that one method wins on arbitrary tasks. Changing the
evaluation seed changes tie-breaking; it does not introduce stochastic roads.

**Your project:** add an obstacle or a stochastic slip. Define its consequences
before training, then compare learning over at least five training seeds. Keep
hazard cost, success rate, episode length, and reward separate in your report.

## Weeks 5–10: change one part of the courier learner

You now have a working table baseline. Extend one component at a time using the
weekly projects, rather than replacing every part at once.

| Week | Small extension | Before running, predict… | Check the result with… |
|---|---|---|---|
| 5 | Encode `(x,y)` and replace the table with a small Q-network | Which output corresponds to each action? | One batch's shapes and a terminal target from exercise 3 |
| 6 | Add just Double Q, or just prioritized replay | Which error is this component intended to reduce? | Same environment and training budget, several seeds |
| 7 | Learn four action probabilities instead of four action values | How should a better-than-average action's probability move? | Browser [**5 · Change the decision rule**](../interactive/index.html#policy), then a two-step finite-difference check |
| 8 | Add a critic and calculate two TD residuals by hand | What changes when the next-value estimate is wrong? | The same MC/TD example before adding GAE |
| 9 | Reuse a batch and limit policy changes with PPO | Which sign of advantage makes each clipping branch active? | Browser probability-ratio controls and the week's scalar exercises |
| 10 | Turn motion into a continuous steering action | What does entropy reward beyond task reward? | A tiny continuous weekly environment; keep the grid baseline for comparison |

The grid has discrete actions, so SAC/TD3's continuous-control implementation
needs a changed action space. Reading its topic does not make the discrete demo
a reproduction of SAC or TD3.

## Week 11: build a study-session bandit with honest logs

This project asks a smaller, useful design question: “Given a context, which
activity should a scheduler suggest?” The simulator uses **fresh/tired** context
and **quiz/read/explain** actions. Its rewards are invented binary outcomes.
It does not claim that these activities have those effects on a real learner.
There is no persistent knowledge state or action-dependent fatigue here; adding
those consequences turns the design into an MDP.

```bash
python -m learning_projects.study_bandit --learner --rounds 1500 --seed 7
```

This uses your week 1 `mean_update`. The same update now keeps a separate mean
for each context/action pair. The simulator's best action is quiz when fresh
and read when tired; their invented means are 0.85 and 0.75, so the oracle target
value under equally likely contexts is `(0.85 + 0.75) / 2 = 0.8`.

The alternative evaluation policy is fixed before collection. Each row stores
the full logger distribution and observed reward, including exploitation and
exploration. Fresh evaluation uses new simulator outcomes after learning.
Compare seeds 1, 7, and 19; a lucky single reward mean does not establish a winner.
The logged reward mean describes the logger's actual choices, whereas IPS and
SNIPS estimate the fixed target's choices. They need not agree on a finite log.
Effective sample size (ESS) describes how concentrated the weights are; it is
not a count of newly created observations or proof that the assumptions hold.

Use [**6 · Reuse old route logs**](../interactive/index.html#ope), then implement `ips`.
The browser starts with two routes and one context; this Python project adds
a separate action distribution for each fresh/tired context:

```bash
python -m learning_projects.check_exercises --exercise 4
```

<details>
<summary>Exercise 4 hint ladder — a weight belongs to an observed row</summary>

<details>
<summary>Hint 1</summary>

an action the logger rarely takes needs more weight when the target
would take it often. The ratio is target propensity divided by logger propensity.

</details>

<details>
<summary>Hint 2</summary>

multiply each observed reward by that row's ratio, then average over
the number of rows. This is IPS; dividing by the sum of weights instead is SNIPS.

</details>

<details>
<summary>Hint 3</summary>

rewards `[1, 1]`, logged propensities `[0.75, 0.25]`, and target
propensities `[0.5, 0.5]` give IPS `(2/3 + 2) / 2 = 4/3`. IPS can exceed the
binary reward range on a finite sample. SNIPS gives 1 for these rows.

</details>

**Transfer:** this exercise assumes full support was checked separately.
No observed-row formula discovers outcomes of a never-tried action. The project
evaluator checks the full logging distribution and refuses missing support.

</details>

**Your project:** make a local habit-dashboard prototype using simulated data
first. Export activity counts, propensities, and a fixed-policy evaluation.
If you later log your own practice sessions, separate measured outcomes from
assumptions; do not infer long-term learning effects from an immediate click.

## Weeks 12–21: grow the project by changing the question

Choose one extension at a time. These are project briefs, not claims that the
small supplied demos already implement all these algorithms.

| Weeks | Project idea | What changes from the baseline? | A useful exercise |
|---|---|---|---|
| 12 | Schedule two parcels for two couriers | Choosing one assignment consumes capacity for another | Enumerate the four possible pairs; distinguish ranking scores from the feasible assignment |
| 13 | Two-player courier or a tiny grid race | The other learner changes your environment | Run a fixed opponent first, then alternating updates; track opponent versions |
| 14–15 | Train a route selector from saved journeys | You cannot query unlogged routes | Delete one route from the data and compare estimates with an unsupported-action warning |
| 16 | Learn a simulator for a road with slips | Planning uses estimated transitions | Browser [**8 · Distrust a perfect-looking plan**](../interactive/index.html#model): raise the error and planning horizon, then compare predicted and real rollouts |
| 17 | Map exploration with a limited battery | Novelty can help discovery and distract from delivery | Plot unique cells visited and delivered parcels separately |
| 18–20 | Rank short route explanations by preference | The label compares outputs rather than giving every action a numeric reward | Browser [**9 · Reward the right behavior**](../interactive/index.html#preferences), then use the real [LLM lab](../llm-lab/README.md) for SFT/DPO |
| 21 | A courier earns points for “delivered” reports | A proxy can be improved without completing delivery | Add a verifier that checks destination state; report proxy score and actual completion |

For language models, the “actions” are tokens and their probabilities are
conditioned on the prefix. Use the two-action policy example to understand the
gradient first; then add sequences, a reference model, and the lab's data splits.
The browser's tiny preference calculation does not train a pretrained LLM.

## Weeks 22–23: separate a penalty from a hard action rule

Try a deliberately bad reward design. The hazard is still a failure with cost 1,
but the objective now pays 5 for entering it:

```bash
python -m learning_projects.courier --hazard-reward 5 --seed 7
python -m learning_projects.courier --hazard-reward 5 --shield --seed 7
```

**Predict:** which policy earns more objective reward? Which actually delivers?
The first learner can chase the hazardous reward. The shield blocks known hazard
entry, including during exploration. A reward penalty changes preferences;
a shield changes which actions are available.

Implement `shield`, preserving the order of permitted actions:

```bash
python -m learning_projects.check_exercises --exercise 5
```

<details>
<summary>Exercise 5 hint ladder — an empty safe set is information</summary>

<details>
<summary>Hint 1</summary>

each candidate action has a predicted immediate cost. Retain costs
less than or equal to the allowed budget; zero budget permits zero cost.

</details>

<details>
<summary>Hint 2</summary>

filter the action list using `costs[action] <= budget`.

</details>

<details>
<summary>Hint 3</summary>

if every candidate is too costly, return an empty list. Do not silently
pick a prohibited action. The caller must define a justified fallback or stop.

</details>

**Transfer:** this teaching shield knows the exact one-step map. A locally safe
action can still lead to future danger in a different environment. Change the
slip model and state exactly what the shield can still guarantee.

</details>

Use [**7 · Price and prevent hazards**](../interactive/index.html#safety) to inspect reward and cost separately.
For a cumulative budget, add remaining budget to the state and test whether
your rule preserves feasibility along a whole trajectory. An average-cost
constraint also differs from a per-episode or worst-case guarantee.

## Weeks 24–26: finish a project someone can inspect

Build a **safe dispatch desk**: two couriers, a handful of parcels, a fixed
capacity, synthetic journey logs, and one known hazard. Keep the interface small:
choose a policy, inspect a route, and see completion, reward, cost, and logging
coverage side by side. Reuse the earlier bandit, courier, and support checks.

For week 25, write an audit table: what data each decision sees, what it changes,
which propensity is logged, what is evaluated on fresh data, and which constraint
is actually enforced. A plausible story is not evidence that a policy is safe.

For week 26, add a simulated “road condition” that affects both route choice and
delivery outcome. Use [**10 · Ask what the logs hide**](../interactive/index.html#causal) to compare an observed
association with a comparison standardized to the same difficulty mix. That
adjustment has a causal interpretation only under stated assumptions, including
no unmeasured confounding, consistency, and overlap; the widget is not an
interventional experiment. The biological extension uses the same
identification questions, but requires domain-specific variables and assumptions;
the courier simulation does not establish a biological causal claim.

Your finished project should include:

- A five-sentence problem statement: state/context, actions, reward, cost, and
  whether actions change future opportunities.
- One hand-calculated update and a test that would fail for a wrong terminal mask.
- A baseline and a modest change measured across several seeds with the same budget.
- Separate training, policy selection, and final evaluation data where appropriate.
- A graph with axes, units, seed count, and an explanation of what the reader can
  conclude from it. Show variability rather than only the best run.
- One failure demonstration, such as missing support, a wrong simulator, a
  misleading proxy, or a shield with an incorrect assumption.

## Check what you learned

Before opening [reference.py](../learning_projects/reference.py), explain one
prediction that was wrong and identify the assumption that caused it. Passing
a small checker is feedback; changing a problem correctly is the transfer test.

After completing the relevant exercises, run:

```bash
python -m learning_projects.check_exercises
```

For an explicit reference check and the project's functional tests:

```bash
python -m learning_projects.check_exercises --reference
python -m unittest discover -s learning_projects/tests -v
```

Those commands have different purposes: the first checks supplied answers;
the second checks the delivered project behavior. They do not complete your
exercise functions. Return to the [progress checklist](PROGRESS.md) and mark
only the calculations and explanations you can now reproduce independently.
