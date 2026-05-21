# STUDY-RL

A 24-week self-paced curriculum in **Deep Reinforcement Learning**, with deep dives into **LLM RL** (RLHF, DPO, GRPO, RLVR, PRMs), **safety-critical RL** (CMDP, shielding, scalable oversight), and an applied thread on **matching/ranking** (insurance leads, recruitment candidates).

> Strong applied-math depth · Tufte-flavored LaTeX notes · Jupyter workbooks with graded exercises · Curated extended reading · Mini-projects with tests.

## Layout

```
STUDY-RL/
├── syllabus/              # syllabus.tex → syllabus.pdf
├── common/                # shared LaTeX style, RL utilities
├── research/              # SOTA dossiers (input to syllabus)
├── week-NN-topic/
│   ├── notes.tex          # lecture notes
│   ├── notes.pdf
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

Requires: TeX Live (with `minted`, so compile with `-shell-escape`), Python 3.11+, CUDA-enabled PyTorch.

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
| Safety-critical RL | 22–24 | CMDPs, CPO, shielding, scalable oversight, reward hacking |

## Reading order

Each week is self-contained; do them in order. Each lecture has a **Safety angle** callout and (where natural) a **Matching application** callout connecting the lesson to insurance lead matching or recruitment candidate matching.
