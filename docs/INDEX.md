# STUDY-RL — Master Index

A 24-week core curriculum, an applied audit in week 25, and a causal-biology extension in week 26.

## Quick links

- **Visual experiments and coding games**: [courier lab](../interactive/index.html) · [activities and hint ladders](INTERACTIVE_LEARNING.md) · [26-week connections](LEARNING_PATH.md)
- **Adversarial learner review**: [rubric, findings and revised scores](LEARNER_REVIEW.md)
- **Start here**: [study guide and all 26 weekly teaching guides](STUDY_GUIDE.md)
- **Current release checks**: [release preparation review](RELEASE_REVIEW.md)
- **Causal biology**: [one-week lesson](../week-26-causal-discovery-biological-data/teaching_guide.md) · [research and source notes](../research/causal_biology.md)
- **Practical LLM post-training**: [LoRA, QLoRA, DPO and quantisation lab](../llm-lab/README.md) · [RTX 5090 setup](../llm-lab/NVIDIA.md)
- **Earlier teaching review (6 October snapshot)**: [archived teaching review](archive/TEACHING_REVIEW.md)
- **Earlier teaching validation**: [archived per-week results](archive/teaching_validation.tsv)
- **Notation and common distinctions**: [glossary](GLOSSARY.md)
- **Syllabus PDF**: [`syllabus.pdf`](../syllabus/syllabus.pdf)
- **Repo README**: [`README.md`](../README.md)
- **Per-week QA report**: [`qa_report.tsv`](archive/qa_report.tsv)
- **PDF margin report**: [`pdf_margin_report.tsv`](archive/pdf_margin_report.tsv)
- **Bundle spec (rules every week follows)**: [`WEEK_BUNDLE_SPEC.md`](WEEK_BUNDLE_SPEC.md)
- **Research dossiers (SOTA inputs)**: [`research/`](../research/)

## Tracks at a glance

| # | Track | Weeks | Highlights |
|---|---|---|---|
| T1 | Foundations | 1–4 | MDPs, bandits, Bellman, DP, MC, TD/SARSA/Q-learning |
| T2 | Deep value | 5–6 | DQN, Double/Dueling/PER, Rainbow, BBF, distributional RL |
| T3 | Policy methods | 7–10 | REINFORCE, GAE, TRPO, PPO, SAC, TD3, CrossQ |
| T4 | Bandits & matching | 11–12 | LinUCB/LinTS, OPE (IPS/SNIPS/DR), CLTR, fairness, two-sided markets |
| T5 | Multi-agent | 13 | MAPPO, MADDPG, QMIX, AlphaZero, PSRO |
| T6 | Offline RL | 14–15 | CQL, IQL, AWR, TD3+BC, Decision Transformer, diffusion policies |
| T7 | Model-based | 16 | Dreamer V3, MuZero, EfficientZero V2, TD-MPC2, DIAMOND |
| T8 | Exploration + meta | 17 | RND, ICM, NGU, MAML, PEARL, in-context RL (AD/DPT) |
| T9 | LLM RL | 18–21 | RLHF/PPO, DPO/IPO/KTO/ORPO/SimPO, GRPO/RLOO/RLVR, PRMs, reward hacking + oversight |
| T10 | Safety-critical RL | 22–24 | CMDPs, CPO, PPO-Lagrangian, shielding, formal methods, capstone |
| T11 | Applied audit | 25 | Synthetic ATS experiments and application audit + RL enhancement roadmap |
| T12 | Causal biology | 26 | Confounding, identification, discovery across environments, task-aware data enhancement |

## Per-week artifacts

Each `week-NN-slug/` contains these learning materials:

- `teaching_guide.md` — prerequisites, analogy boundaries, derivation, numerical examples, code bridge, and self-checks
- `notes.pdf` — Tufte-flavored lecture with derivations + figures
- `extended_reading.pdf` — annotated focused reading
- `workbook.ipynb` — exercises with inline hints and reference implementations; `.executed.ipynb` is generated locally and excluded from Git
- `project/` + `tests/` — runnable mini-project with pytest smoke + solution tests

## Working with the curriculum

```bash
# Setup from the repository root
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install 'swig==4.*'
python -m pip install -r requirements.txt

# Build everything
make course-tests PY=python  # every weekly suite and shared tools
make release-check PY=python
bash scripts/qa_sweep.sh  # per-week status table

# Per-week loop
cd week-NN-slug
latexmk -pdf -shell-escape notes.tex     # rebuild PDF
python -m jupyterlab workbook.ipynb      # select the course kernel
PYTHONPATH=../ pytest tests -q           # check
```

## Maintenance scripts

| Script | What it does |
|---|---|
| `scripts/new_week.py` | Scaffold a new week-NN-slug/ folder |
| `scripts/qa_sweep.sh` | Inventory artifacts and run each week's tests, write `reports/qa_report.tsv` (does not compile or execute workbooks) |
| `scripts/pdf_margin_sweep.py` | Scan all PDFs for overfull-hbox + page-edge cutoffs |
| `scripts/recompile_all_pdfs.sh` | Idempotent rebuild of every PDF |
| `scripts/append_hints_section.py` | Consolidate exercise-specific hints; `--refresh` replaces the generated section |

## Validation and authoring

File presence and archived results do not establish a fresh execution check.
The [release review](RELEASE_REVIEW.md) records public-release validation; the
[teaching-review archive](archive/TEACHING_REVIEW.md) preserves earlier corrections
and evidence. Follow the [README](../README.md) for the complete installation,
optional lab, PDF build, and notebook execution paths.

The authoring requirements and scaffold are documented in
[WEEK_BUNDLE_SPEC.md](WEEK_BUNDLE_SPEC.md) and [scripts/new_week.py](../scripts/new_week.py).

The current teaching pass adds a guide for every week and corrects mathematical
and explanatory errors in the lecture sources. The guides explicitly identify
where mini-projects simplify the algorithms discussed in the papers.
Week 26 extends the applied curriculum with causal biology; its research dossier
distinguishes peer-reviewed methods, public event descriptions, and proprietary
company claims.

## Conventions

- All Python imports work from the repo root with `PYTHONPATH=.`.
- Weekly test suites run in separate processes via `make tests`; combining
  several weeks in one pytest invocation is rejected to prevent package collisions.
- All seeds default to `0` (see `common/seed.py`).
- All plots use `common/plotting.use_rl_style()` and `RL_PALETTE`.
- LaTeX must compile with `-shell-escape` (minted dependency).
- Every week has a `safetyangle` callout and a `matching` callout linking back to synthetic insurance matching or synthetic ATS (recruitment).
