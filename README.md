# STUDY-RL

A 24-week core curriculum, plus an applied audit in week 25 and **causal discovery for biological data in week 26**, in **Deep Reinforcement Learning**, with deep dives into **LLM RL** (RLHF, DPO, GRPO, RLVR, PRMs), **safety-critical RL** (CMDP, shielding, scalable oversight), and an applied thread on **matching/ranking** (insurance leads, recruitment candidates).

> Strong applied-math depth · Tufte-flavored LaTeX notes · Jupyter workbooks with graded exercises · Curated extended reading · Mini-projects with tests.

**Start with [the study guide](STUDY_GUIDE.md):** a prerequisite check, one small
problem connecting Bellman equations to policy gradients, and links to all 26
weekly teaching guides. Each weekly guide adds worked calculations, an analogy
with its limits, a derivation, code connections, and hidden-answer self-checks.
The [teaching review](TEACHING_REVIEW.md) records corrections and remaining limitations.

**Causal biology:** [Week 26](week-26-causal-discovery-biological-data/teaching_guide.md)
provides a one-week plan on confounding, causal identification and discovery,
learning across biological environments, and task-aware enhancement of noisy,
sparse data. It builds on weeks 11 and 14, with synthetic patient experiments
and a [primary-source research dossier](research/causal_biology.md) inspired by
Synlico's scientific discussion.

**Practical LLM training:** the [five-lesson post-training lab](llm-lab/README.md)
adds actual pretrained-model LoRA SFT and DPO, a quantisation workbook, saved
adapter evaluation, and [RTX 5090 / NVIDIA QLoRA setup](llm-lab/NVIDIA.md).

## Layout

```
STUDY-RL/
├── syllabus/              # syllabus.tex → syllabus.pdf
├── common/                # shared LaTeX style, RL utilities
├── research/              # SOTA dossiers (input to syllabus)
├── llm-lab/               # practical LoRA, QLoRA, DPO, quantisation, evaluation
├── week-NN-topic/
│   ├── notes.tex          # lecture notes
│   ├── notes.pdf
│   ├── teaching_guide.md  # intuition, derivation, worked examples, self-checks
│   ├── workbook.ipynb     # graded exercises with hints
│   ├── extended_reading.tex
│   ├── project/           # mini-project code
│   └── tests/             # pytest unit + smoke tests
└── scripts/               # QA, plotting, helpers
```

## Build

```bash
pip install -r requirements.txt
make syllabus        # build the top-level syllabus PDF
make weeks           # compile all weekly lecture PDFs
make notebooks       # execute every workbook end-to-end via papermill
make tests           # run all pytest suites
make qa              # full QA sweep + report
```

`make tests` runs each weekly suite in its own Python process, then the shared
tooling and optional lab tests. Bare `python -m pytest` checks shared tooling and
the lab. Use `python -m pytest week-NN-topic/tests` for one week; collecting
several weeks together is rejected because their local `project` and `tests`
package names can collide.

Requires: TeX Live (with `minted`, so compile with `-shell-escape`) for PDF
builds, and Python with the dependencies needed by the selected project. The
small numerical examples and many toy projects run on CPU; CUDA is useful for
larger deep-RL experiments. The full requirements file includes optional heavy
environment and LLM dependencies and may need platform-specific installation.

## Curriculum at a glance

The full week-by-week breakdown lives in [`syllabus/syllabus.pdf`](syllabus/syllabus.pdf). Tracks:

| Track | Weeks | Highlights |
|---|---|---|
| Foundations | 1–4 | MDPs, Bellman, DP, MC, TD/SARSA/Q-learning |
| Function approximation | 5–6 | Deep Q-Networks, Rainbow, BBF |
| Policy methods | 7–10 | REINFORCE, A2C, GAE, PPO, TRPO, SAC, TD3 |
| Bandits & matching | 11–12 | LinUCB, Thompson, OPE, counterfactual LTR |
| Multi-agent | 13 | MADDPG, MAPPO, self-play |
| Offline RL | 14–15 | CQL, IQL, AWR, Decision Transformer, diffusion policies |
| Model-based RL | 16 | Dreamer v3, MuZero, TD-MPC2 |
| Exploration & meta | 17 | RND, ICM, MAML, in-context RL |
| LLM RL | 18–21 | RLHF/PPO, DPO/IPO/KTO/ORPO/SimPO, GRPO/RLOO/RLVR, PRMs |
| Practical LLM lab | Alongside 18–21 | Real LoRA SFT/DPO, quantisation/QLoRA, Mac + RTX 5090 workflows |
| Safety-critical RL | 22–24 | CMDPs, CPO, shielding, scalable oversight, reward hacking |
| Applied audit | 25 | Synthetic ATS experiments and an RL enhancement audit |
| Causal biology | 26 | DAGs, confounding, multi-environment discovery, noisy data, enhancement evaluation |

## Reading order

Follow the core weeks in order: later derivations reuse earlier results. Read
the weekly teaching guide before the longer lecture, attempt the workbook, then
run the mini-project. Each lecture has a **Safety angle** callout and (where
natural) a **Matching application** connecting the lesson to insurance lead or
recruitment candidate matching. Track both understanding and execution in
[PROGRESS.md](PROGRESS.md); running a reference solution alone does not demonstrate mastery.
