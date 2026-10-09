# STUDY-RL

A self-paced reinforcement-learning course with a 24-week core, a synthetic
recruitment-system audit in week 25, and a causal-biology extension in week 26.
Each week includes a teaching guide, lecture PDF and TeX source, a Jupyter
workbook, extended reading, a small Python project, and tests. An optional
five-lesson lab trains real pretrained language models with LoRA and DPO.

**Start here:** [study guide](docs/STUDY_GUIDE.md) ·
[interactive learning path](docs/LEARNING_PATH.md) ·
[all weeks and artifacts](docs/INDEX.md) ·
[syllabus PDF](syllabus/syllabus.pdf) ·
[glossary](docs/GLOSSARY.md) · [progress checklist](docs/PROGRESS.md).

Begin with Python, averages and basic probability. Linear algebra, calculus and
neural networks become useful later; the [learning path](docs/LEARNING_PATH.md)
introduces prerequisites when each track needs them. Start with week 1 if RL is
new to you; follow the track table below if you already know the foundations.

## Learn by playing and changing a number

The [courier learning lab](interactive/index.html) has ten staged experiments:
route exploration, Bellman backups, MC/TD targets, a learning grid game, policy
gradients/PPO/GAE, old-policy logs, cost budgets, model error, preferences, and
confounding. Graphs and animations show actual simulated decisions or named
exact formulas. Each experiment asks you to predict, inspect, and change one
assumption. The browser lab needs no packages or internet connection.

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Visit **http://localhost:8000/interactive/**. Serving the root keeps course links
working. `make lab PY=python` does the same; Ctrl+C stops the server. You can also
open `interactive/index.html` directly. Start with **1 · Try a route** and save
later experiments for their weeks.

The [coding walkthrough](docs/INTERACTIVE_LEARNING.md) gives hint ladders and
small projects: play a courier game, train its policy, simulate a study-activity
recommender, evaluate logs, and add a hazard shield. They use Python's standard
library. Five learner exercises begin as real TODOs; their checker reports
unfinished or incorrect work without substituting reference answers:

```bash
python -m learning_projects.courier --play
python -m learning_projects.check_exercises --exercise 1
```

The second command intentionally reports `PENDING` until you implement
`mean_update` in `learning_projects/exercises.py`. Broader game, planning,
ranking and LLM project ideas appear in the walkthrough. These toy experiments
teach mechanisms; the existing weekly projects and readings develop the full
algorithms. See the [adversarial learner review](docs/LEARNER_REVIEW.md) for the
review rubric and [lab development checks](interactive/README.md) for browser tests.

## Read without installing anything

Open a week's `teaching_guide.md`, then `notes.pdf`, and preview its
`workbook.ipynb` on GitHub. PDFs are committed so reading does not require
Python, a GPU, or a LaTeX installation. The source notebooks contain exercises
and reference-solution paths; attempting the exercises yourself is separate
from executing the reference notebook.

## Install the course environment

Python **3.12** is recommended. Run these commands from a terminal on macOS or
Linux. On Windows, use PowerShell for activation (`.venv\Scripts\Activate.ps1`);
the Makefile and shell scripts can be run through WSL.

```bash
git clone https://github.com/samuel-zhang01/STUDY-RL.git
cd STUDY-RL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'swig==4.*'
python -m pip install -r requirements.txt
python -m ipykernel install --user --name study-rl --display-name "STUDY-RL"
```

`requirements.txt` installs the scientific stack, CPU-capable PyTorch,
classic-control and Box2D Gymnasium environments, notebook tools, and test tools for the
bundled projects. It does not install every optional paper benchmark or the
pretrained LLM lab. If you need an NVIDIA-specific PyTorch build, install the
wheel for your hardware first, following [the NVIDIA guide](llm-lab/NVIDIA.md).

Box2D is needed by the LunarLander DQN/Rainbow exercises. Install SWIG before
the requirements so pip can build its Python bindings when no compatible
wheel is available. A native build also needs a C++ compiler (Xcode Command
Line Tools on macOS, or the platform's compiler toolchain). See
[Gymnasium's Box2D installation notes](https://gymnasium.farama.org/environments/box2d/).

Verify the installation with a small first-week check:

```bash
python -m pytest common/tests -q
python -m pytest week-01-mdps-bandits/tests -q
```

Dependencies have bounded version ranges, rather than a platform-independent
lockfile. The [release review](docs/RELEASE_REVIEW.md) records the environment
actually checked. The optional benchmark collection is in
`requirements-extras.txt`; install it only for an experiment that needs it.
Some environments require additional system libraries and separately obtained
assets. The repository does not bundle Atari ROMs or accept their terms for you.

## Work through your first week

1. Read [week 1's teaching guide](week-01-mdps-bandits/teaching_guide.md).
   Predict the worked-example answers before revealing them.
2. Read [the lecture](week-01-mdps-bandits/notes.pdf) and use the
   [glossary](docs/GLOSSARY.md) when notation is unfamiliar.
3. Open the workbook from its weekly directory:

   ```bash
   cd week-01-mdps-bandits
   python -m jupyterlab workbook.ipynb
   ```

   Select the **STUDY-RL** kernel. Start with a fresh kernel and keep the
   notebook's working directory inside that week; the imports assume this.
   Save a separate local copy for your own attempts. The distributed notebook
   can execute using reference solutions in `project/solutions.py`.
4. Inspect and run the small bandit experiment from the same weekly directory:

   ```bash
   python -m project.train
   python -m pytest tests -q
   cd ..
   ```

5. Record what you can explain, derive, and reproduce in your own copy of the
   [progress checklist](docs/PROGRESS.md). Continue with week 2.

For a noninteractive reference execution from the repository root:

```bash
python -m papermill --kernel study-rl --cwd week-01-mdps-bandits \
  week-01-mdps-bandits/workbook.ipynb \
  week-01-mdps-bandits/workbook.executed.ipynb
```

The output file is ignored by Git. Other weekly projects follow the same
directory pattern, though their command-line options differ; inspect the
module docstring or `--help` where provided. Larger optional training runs are
not required to read the course or run the local reference exercises.

## Choose a track

| Track | Weeks | Main topics |
|---|---|---|
| Foundations | 1–4 | MDPs, bandits, Bellman equations, DP, Monte Carlo, TD, SARSA, Q-learning |
| Deep value learning | 5–6 | DQN, replay, target networks, Rainbow components, BBF readings |
| Policy methods | 7–10 | REINFORCE, A2C, GAE, TRPO, PPO, SAC, TD3 |
| Bandits and matching | 11–12 | LinUCB/Thompson sampling, off-policy evaluation, ranking, matching |
| Multi-agent | 13 | Coordination, policy learning, self-play, AlphaZero-style search |
| Offline RL | 14–15 | CQL, IQL, sequence models, Decision Transformer, diffusion readings |
| Model-based RL | 16 | Learned dynamics, imagination, planning |
| Exploration and meta-learning | 17 | Intrinsic rewards, RND, meta-bandit experiments |
| LLM RL | 18–21 | RLHF/PPO, preference objectives, GRPO/RLOO/RLVR, reward hacking, oversight |
| Safety and capstone | 22–24 | CMDPs, Lagrangian methods, shielding, safe matching |
| Applied audit | 25 | Synthetic ATS policies, propensities, OPE, constraints, audit design |
| Causal biology | 26 | Confounding, causal identification, environment shifts, noisy data |
| Practical LLM lab | Alongside 18–21 | Real LoRA SFT/DPO, quantisation, adapter evaluation, QLoRA setup |

The [curriculum index](docs/INDEX.md) links every weekly artifact. Paper names
in the notes describe reading topics; the bundled teaching projects are small
implementations and do not reproduce every named paper or benchmark.

## Optional practical LLM lab

Use a separate environment to keep its tested model-library versions apart
from optional RL benchmark dependencies:

```bash
python3.12 -m venv .venv-llm
source .venv-llm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r llm-lab/requirements.txt
make llm-tests PY=python
make llm-notebook PY=python
cd llm-lab
python -m llm_lab.doctor
```

The numerical workbook and tests use local tensors and random tiny models;
they do not download pretrained weights. The optional training exercises
download `HuggingFaceTB/SmolLM2-135M-Instruct` from Hugging Face and train
locally. CPU and Apple Silicon support ordinary LoRA; NF4 QLoRA targets NVIDIA
CUDA. No paid API is required. Follow [the full lab walkthrough](llm-lab/README.md)
for data generation, SFT, DPO, saved-adapter inference, and held-out evaluation.
See [recorded lab validation](llm-lab/VALIDATION.md) for measured results and
[the NVIDIA guide](llm-lab/NVIDIA.md) for CUDA installation and hardware checks.

## Test, execute, and rebuild

Run Make targets from the repository root with the intended environment active.
`make tests` includes Node.js simulation checks; `make course-tests` and
`make learning-tests` only need Python. Browser checks additionally require
the separate development setup in [the lab README](interactive/README.md):

| Command | What it does |
|---|---|
| `make help` | List common tasks |
| `make course-tests PY=python` | Shared utilities, every weekly suite in its own process, maintenance scripts |
| `make llm-tests PY=python` | Optional lab tests; requires lab dependencies |
| `make tests PY=python` | Course, optional LLM lab, Python learning projects and Node simulation checks |
| `make learning-tests PY=python` | Dependency-free courier, study-bandit and exercise checks |
| `make lab-tests` | Numerical browser simulation checks; requires Node.js |
| `make browser-tests PY=python` | Full learning-lab workflows in Chromium, Firefox and WebKit; see lab setup |
| `make lab PY=python` | Serve the local visual lab and course guides |
| `make release-check PY=python` | File hygiene, Python/notebook structure, local documentation links |
| `make secrets` | Redacted Gitleaks scan of all reachable Git refs; install Gitleaks separately |
| `make syllabus` | Rebuild `syllabus/syllabus.pdf` |
| `make weeks` | Rebuild all weekly lecture and extended-reading PDFs |
| `make notebooks` | Execute all weekly reference notebooks using the `python3` kernel |
| `make llm-notebook PY=python` | Execute the optional lab's numerical workbook |
| `make qa PY=python` | Build PDFs, execute weekly notebooks, run all tests, then inventory artifacts |
| `make clean PY=python` | Remove ignored build caches and executed notebooks; preserve shipped PDFs and environments |

For `make notebooks`, ensure the `python3` kernel uses the active course
environment: `python -m ipykernel install --user --name python3`.
Bare `python -m pytest` covers shared tools and the optional lab. Weekly bundles
reuse `project` and `tests` package names, so run one week per Python process;
combined collection of multiple weeks is intentionally rejected.

PDF rebuilding requires TeX Live or MacTeX, `latexmk`, and the packages used by
`common/rlstudy.sty`, including `minted` and its configured Python syntax
highlighter. The build enables shell escape. To rebuild one week:

```bash
cd week-01-mdps-bandits
latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error notes.tex
latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error extended_reading.tex
cd ..
```

To inspect all rebuilt PDFs for layout problems:

```bash
python scripts/pdf_margin_sweep.py --render
```

This requires Poppler (`pdftoppm`, `pdfinfo`) and Pillow. Fresh TSV reports are
written under ignored `reports/`; earlier records are in
[the validation archive](docs/archive/README.md).

## Repository layout

```text
STUDY-RL/
├── README.md, LICENSE, CONTRIBUTING.md, SECURITY.md
├── docs/                 # study guide, index, glossary, progress, authoring spec
│   └── archive/          # dated teaching reviews and earlier validation records
├── syllabus/             # syllabus.tex and the canonical syllabus.pdf
├── common/               # shared Python utilities, tests, and LaTeX style
├── interactive/          # offline visual learning lab; development browser tests
├── learning_projects/    # standard-library games, learner TODOs, references
├── research/             # dated source dossiers and reading notes
├── llm-lab/              # optional LoRA/DPO lab, lessons, synthetic fixtures
├── week-NN-topic/
│   ├── teaching_guide.md
│   ├── notes.tex, notes.pdf
│   ├── workbook.ipynb
│   ├── extended_reading.tex, extended_reading.pdf
│   ├── project/          # teaching implementation and reference solutions
│   ├── tests/
│   └── figs/             # figures used by the notes
├── scripts/              # scaffolding, validation, and cleanup tools
└── .github/workflows/    # CPU tests, repository checks, and history secret scan
```

## Troubleshooting

- **`No module named common` or the wrong `project` imports:** run a workbook
  or `python -m project.train` inside its weekly directory. Run a selected test
  suite from the repository root or use the Make targets. Restart the kernel
  when switching weeks.
- **Notebook cannot find its kernel:** activate the environment and register
  it with `python -m ipykernel install`; select that kernel in JupyterLab.
- **Missing environment or CUDA package:** the core install supports bundled
  teaching projects. Use the optional extras only when required and follow
  the hardware-specific guide for the PyTorch build.
- **`Box2D is not installed` or a Box2D build fails:** install SWIG first with
  `python -m pip install 'swig==4.*'`, ensure a C++ compiler is available, then
  rerun `python -m pip install -r requirements.txt`.
- **PDF build reports `minted`/highlighter errors:** check TeX Live and the
  highlighter used by its `minted` version. Read the committed PDFs while
  fixing the local build setup.
- **A trainer refuses an existing output directory:** use a new `runs/`
  subdirectory; lab commands protect previous adapters from being overwritten.

## Scope, contributions, and license

Experiments use synthetic teaching fixtures. The recruitment examples are
hypothetical, and the biology extension contains no real patient records.
Toy outcomes do not establish production safety, clinical effectiveness, or
general model quality. Bibliographies and research dossiers are dated reading
material, not a claim to cover the latest literature.

See [the release review](docs/RELEASE_REVIEW.md) for checked paths and remaining
limits, [contribution instructions](CONTRIBUTING.md) for changes, and
[security reporting](SECURITY.md) for sensitive findings.

Original contributions are distributed under the [MIT license](LICENSE).
Cited papers, external dependencies, downloaded models, and trademarks retain
their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
