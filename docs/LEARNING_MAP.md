# Choose your route

[Study guide](STUDY_GUIDE.md) · [Visual explanations](CONCEPT_ATLAS.md) · [Interactive lab](../study-lab/README.md) · [Progress](PROGRESS.md) · [Glossary](GLOSSARY.md)

**New to RL? Start with weeks 1–4.** You do not need a GPU, a LaTeX
installation, or all the packages in `requirements.txt` to begin. Read the
teaching guides and existing PDFs in your repository viewer. The interactive
experiments run in a browser. The actual exercises live in Jupyter notebooks
and Python files on [GitHub: samuel-zhang01/STUDY-RL](https://github.com/samuel-zhang01/STUDY-RL).
Use the website to learn and explore, then keep your own working copy of the
materials for coding.

## Your first 30 minutes

1. **5 minutes:** attempt only the weighted-average and conditional-probability
   [prerequisite questions](STUDY_GUIDE.md#prerequisite-check-five-tools-you-will-reuse).
   Save discounting for week 2 and gradients for week 7.
2. **10 minutes:** open [Try a route](../interactive/index.html#bandit).
   Predict the outcome of five route choices. Each choice reveals one outcome;
   it does not tell you what every other route would have paid.
3. **10 minutes:** use [week 1](../week-01-mdps-bandits/teaching_guide.md) to
   update one arm's mean for observations `(1,0,1)`. Calculate `1, 0.5, 2/3`
   before running the notebook. Explain why the count belongs to that arm.
4. **5 minutes:** write one unanswered question in your progress record and
   choose the next session's [bounded notebook investigation](../week-01-mdps-bandits/teaching_guide.md#notebook-investigation-track-uncertainty-separately-from-reward).

This page handles routes, time budgets and downloads. The
[staged learning path](LEARNING_PATH.md) explains how each new concept grows
from the previous one; [interactive activities and coding projects](INTERACTIVE_LEARNING.md)
give smaller practice steps when a weekly workbook is still too large.

## Routes with explicit prerequisites

Arrows below indicate a recommended reading order. They are not claims that
every topic in a week is a prerequisite for every later topic.

| Your goal | Route | Finish by demonstrating |
|---|---|---|
| Build a foundation | **1 → 2 → 3 → 4**, then 5 → 6 | Compute MC, SARSA and Q-learning targets on the same transition; explain why they differ |
| Rebuild probability and statistics | [Conditioning → expectation → sampling → comparisons](statistics/README.md), alongside the weekly lessons | Explain a conditional mean, choose the independent unit, and interpret an interval without turning it into a guarantee |
| Understand policy optimisation | **1–4 → 7 → 8 → 9**, then 10 | Explain an advantage, GAE boundary and both signs of PPO clipping |
| Understand LLM post-training | **1–4 → 7 → 8 → 9 → 18 → 19 → 20 → 21**; add the [practical lab](../llm-lab/README.md) | Distinguish the rollout policy, reference policy, reward and verifier; evaluate a saved model |
| Learn from logged decisions | **1–4 → 11 → 12**; add **5 → 14 → 15** for offline deep RL | Identify support failures before interpreting an offline estimate |
| Study safety and applied audits | **1–4 → 7–9 → 11 → 22 → 23**; add **5 → 12 → 14 → 24 → 25** | Distinguish an average cost budget, an action filter and evidence for a deployment claim |
| Study causal biology | **1–3 → 11 → 26**; use week 3's importance weighting, week 11's bandit/OPE material and optionally 14's support discussion | Name identification assumptions and separate effect estimates from prediction accuracy |
| Complete the course | **1–24**, then optional applied extensions **25–26** | Produce a project report with baselines, changed assumptions and limitations |

Use the [week directory](STUDY_GUIDE.md#week-by-week-teaching-guides) to open any
number above. Every weekly guide also links to its immediate prerequisites,
lecture PDF, workbook, previous and next week. Sequential navigation follows
course order; your chosen route may skip ahead.

## How much should I do this week?

These are planning budgets, not measured completion times. Training and
installation time are additional and vary by machine.

| Time available | Work to do | Stop condition |
|---|---|---|
| 30 minutes | Opening question, one worked example, one retrieval question | Explain one update and record your uncertainty |
| 2 hours | 20 min guide, 30 min calculations, 45 min workbook attempt, 25 min check and error log | Reproduce a calculation without looking and try a changed input |
| 6–10 hours | Add the relevant lecture sections, independent implementation and a small project evaluation | Report the data, baseline, seed, metric and a failure case |
| Extra depth | Extended reading and larger experiments after the core checks | State exactly which extra claim the experiment tests |

**Core:** the guide, its worked examples, retrieval questions, and one independent
transfer attempt. **Depth:** lecture derivations and additional exercises.
**Extension:** paper reproduction or large training. Reading every paper is not
required before moving to the next week.

Week 26 has its own [five-session plan](../week-26-causal-discovery-biological-data/teaching_guide.md#one-week-five-two-hour-sessions).
For other weeks, pick the lecture sections named in the guide's code bridge
and workbook exercise references.

## If a prerequisite is missing

| Where you get stuck | Repair task | Ready when you can… |
|---|---|---|
| Expectations and probabilities | Rework the weighted-average question, then [week 1's online average](../week-01-mdps-bandits/teaching_guide.md) | Explain why an expectation need not be an observed outcome |
| Discounting and indices | Draw the [reward timeline](CONCEPT_ATLAS.md#reward-return-and-value) | Place the first reward at exponent zero |
| Bellman equations | Follow [the target comparison](CONCEPT_ATLAS.md#one-transition-different-targets), then week 2 | Separate a sampled next state from an average over next states |
| Gradients or logs | Work through [the two-action derivative](STUDY_GUIDE.md#3-advantage-and-policy-gradients-compare-an-action-with-the-policy-average), then week 7 | Derive both action scores, `1−p` and `−p` |
| Tensor shapes | Do the shape question in the prerequisite check; inspect week 5's code bridge | Explain the accidental `[B,B]` broadcast |
| Probability ratios | Start with [logged support](CONCEPT_ATLAS.md#logged-data-and-support), then week 11 | State which policy generated the action and which policy is being evaluated |
| Error bars and comparisons | Use [sampling](statistics/03-sampling-uncertainty.md), then [intervals and comparisons](statistics/04-intervals-comparisons.md) | Distinguish SD, SE, and an interval; identify whether runs are genuinely paired |
| Bootstrap or dependent observations | Read [resampling and dependence](statistics/05-resampling-dependence.md) | Choose the unit to resample and state what is held fixed |
| Implementation | Try [progressive coding exercises](INTERACTIVE_LEARNING.md) before the full project | Pass a changed-input check using your own function |
| Installation | Start with reading, browser experiments and standard-library practice | Separate a missing package from a conceptual error |

## Open the right artifact

| Artifact | Use it for | What completing it establishes |
|---|---|---|
| `teaching_guide.md` | First explanation, calculations and retrieval | A route into the topic |
| `notes.pdf` | Typeset derivations and figures | Deeper reading; no compile step needed |
| `workbook.ipynb` | Exercises and demonstrations | Some cells call reference code; execution alone is not independent assessment |
| `project/` and `tests/` | Study and verify the bundled implementation | Behaviour on those fixtures |
| `extended_reading.pdf` | Follow primary sources | Extra depth and paper context |
| [Progressive coding exercises](INTERACTIVE_LEARNING.md) | Write small functions before viewing answers | Correctness on several unseen boundary cases |

## Download and open the Jupyter materials

**Canonical repository:** [samuel-zhang01/STUDY-RL on GitHub](https://github.com/samuel-zhang01/STUDY-RL).
Choose **Code → Download ZIP**, or [download the complete main-branch ZIP](https://github.com/samuel-zhang01/STUDY-RL/archive/refs/heads/main.zip)
and extract it. Git users can clone it:

```bash
git clone https://github.com/samuel-zhang01/STUDY-RL.git
cd STUDY-RL
```

Keep the whole folder together. A weekly notebook imports both its adjacent
`project/` folder and the repository's shared `common/` package. Downloading
only `workbook.ipynb` is useful for reading or saving an exercise, but does not
make it a standalone runnable project. GitHub renders a preview; it does not
execute the Python cells. The ZIP is a snapshot of a moving branch; record
the commit when comparing results. [GitHub documents archive contents and revision-specific downloads](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives).

Read the repository's [current setup instructions](../README.md) before installing
packages. Use its CPU requirements for the weekly teaching projects; optional
environment and LLM stacks have separate requirements. From the repository root,
create an environment, install those requirements, and launch the week you chose.
For example, on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd week-01-mdps-bandits
python -m jupyterlab workbook.ipynb
```

On Windows, create the environment with `py -m venv .venv`; use
`.venv\Scripts\Activate.ps1` in PowerShell before the remaining commands.
If activation is restricted, invoke the environment's `python.exe` by its full
path instead. Keep Jupyter's selected Python kernel in that same environment.
JupyterLab opens in your browser while Python runs on your computer; see
[the JupyterLab starting guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html).

Save a personal notebook copy before editing. Start with the **bounded first
investigation** near its top, then:

1. Read the prompt and write a prediction without running the solution cell.
2. Read the mode instructions at the top. In scaffolded workbooks,
   `REFERENCE_MODE=True` demonstrates the supplied implementation. Complete
   the scaffold, choose `False` in setup, restart, and run the learner checks.
   A guarded reference wrapper must be replaced by your own implementation;
   an unfinished learner raises `NotImplementedError` without a fallback.
3. Run the setup cell, your exercise, and its check in order. A cell may rely
   on variables created earlier; execution numbers reveal the order you used.
4. Change one input as suggested by the guide, predict again, and explain the
   difference. Keep the failed prediction in your notes.
5. Restart the kernel and rerun the required cells from the top. This catches
   results that depended on hidden state left from an earlier attempt.

Scaffolded workbooks explicitly select learner or reference implementations;
Week 26 also keeps its existing no-fallback learner selection. Some other
workbooks contain direct experiments rather than a function scaffold: their
reference calculations remain demonstrations. Read the local instructions,
write a prediction, and attempt the changed-assumption task yourself. Running
every cell in default reference mode is a reproducibility check, not a grade
for independently solving every exercise. Long training cells can wait until
you have completed the small calculation and masking checks.

| Symptom | Check first |
|---|---|
| `No module named common` or `project` | Preserve the repository layout and open the notebook from its week folder |
| A package installed successfully but the notebook cannot import it | Confirm the selected kernel uses the environment where you installed it |
| A changed function seems to have no effect | Rerun its definition cell and confirm the check calls it rather than a reference |
| A result works only after unrelated cells | Restart the kernel and run the required cells in order |
| A plot differs from the guide | Compare fixture, seed, units, terminal masks and reduction convention before assuming either result is wrong |

Run that week's tests from the repository root with
`python -m pytest week-NN-topic/tests -q`, replacing the directory name.
Run separate weeks in separate processes because they reuse module names.
Tests check the bundled implementation and their fixtures; they do not replace
your notebook attempt or establish a performance theorem.

## Resume after a break

Open your last [progress entry](PROGRESS.md). Without the guide, explain its
main idea and redo one calculation. If either fails, reopen that week's worked
example; if both pass, change one assumption before moving on. A checked box
records evidence, not permanent mastery.
