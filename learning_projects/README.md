# Small RL projects you can change

These projects use only the Python standard library. Run every command from the
repository root; Python 3.10 or newer is sufficient for this folder. The full
course recommends Python 3.12 and has additional dependencies.

Start with [the interactive learning walkthrough](../docs/INTERACTIVE_LEARNING.md).
The [learning path](../docs/LEARNING_PATH.md) connects the projects to all 26 weeks.

```bash
python -m learning_projects.two_step
python -m learning_projects.courier --play
python -m learning_projects.courier --episodes 1200 --seed 7
python -m learning_projects.study_bandit --rounds 1500 --seed 7
```

- **Two-step delivery:** reward now versus reward later; then compare MC and TD.
- **Courier game:** move around a 4×4 map, train a Q-table, inspect the actual
  greedy path, and distinguish objective reward from hazard cost.
- **Study-session bandit:** choose quiz/read/explain in two invented contexts,
  record the full logging policy, and evaluate a fixed alternative policy.
  This is a synthetic project, not evidence about anyone's learning habits.

The two-step command starts with returns/values only. Add `--stage targets`
at weeks 3–4 and `--stage policy` at week 7; `--stage all` shows the full comparison.

The demos use reference implementations and label that explicitly. Your work
lives in [exercises.py](exercises.py); its five functions start unfinished and
never call an answer for you. Check only the exercise for your current week:

```bash
python -m learning_projects.check_exercises --exercise 1
```

`PENDING` is expected before editing. It exits with status 1, just like an
incorrect answer. After completing exercise 1, use it in the bandit:

```bash
python -m learning_projects.study_bandit --learner --rounds 1500 --seed 7
```

After exercise 3, use your own target in the courier:

```bash
python -m learning_projects.courier --learner --episodes 1200 --seed 7
```

`--learner` refuses an unfinished function. It does not substitute a reference
answer. Exercises 2, 4, and 5 are checked directly and have extension tasks in
the walkthrough; the project CLIs integrate exercises 1 and 3 only.

For maintainers or after attempting the exercises:

```bash
python -m learning_projects.check_exercises --reference
python -m unittest discover -s learning_projects/tests -v
```

The first command checks supplied answers; it does not grade your work.
The second checks the delivered reference projects, boundaries, logged support,
multi-seed deterministic-map learning, and real CLI workflows. Passing those
tests does not mean you have completed the learner exercises.

Reward-design and shielding experiment:

```bash
python -m learning_projects.courier --hazard-reward 5 --seed 7
python -m learning_projects.courier --hazard-reward 5 --shield --seed 7
```

The first rewards entering a hazard; the second forbids known hazard-entry
actions using an exact one-step map. This shield's assumptions are unusually
strong: unknown dynamics and cumulative risks require additional reasoning.
