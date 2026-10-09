# STUDY-RL — build orchestration

LATEXMK := latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error
WEEKS := $(sort $(wildcard week-*))
PY := python3

.PHONY: help all syllabus weeks notebooks tests course-tests learning-tests lab-tests browser-tests lab llm-tests llm-notebook qa release-check secrets clean cleanall

help:
	@echo "Install: python -m pip install -r requirements.txt"
	@echo "course-tests: shared tools and every weekly suite (CPU)"
	@echo "tests: course-tests plus optional LLM lab tests"
	@echo "lab: open the local interactive learning lab at http://localhost:8000/interactive/"
	@echo "learning-tests / lab-tests: dependency-free Python projects / Node simulation checks"
	@echo "browser-tests: Chromium, Firefox and WebKit interactions (install interactive dev dependencies)"
	@echo "release-check: repository hygiene, notebook structure, local links"
	@echo "secrets: Gitleaks scan of every Git ref (requires gitleaks)"
	@echo "syllabus / weeks: rebuild PDFs (requires TeX Live)"
	@echo "notebooks / llm-notebook: execute reference workbooks"
	@echo "clean: remove generated caches while retaining shipped PDFs"

all: syllabus weeks notebooks tests

syllabus:
	cd syllabus && $(LATEXMK) syllabus.tex

weeks:
	@for w in $(WEEKS); do \
	  echo "=== building $$w ==="; \
	  (cd $$w && $(LATEXMK) notes.tex && $(LATEXMK) extended_reading.tex) || exit 1; \
	done

notebooks:
	@for w in $(WEEKS); do \
	  if [ -f $$w/workbook.ipynb ]; then \
	    echo "=== executing $$w/workbook.ipynb ==="; \
	    papermill --kernel python3 --cwd "$$w" $$w/workbook.ipynb $$w/workbook.executed.ipynb || exit 1; \
	  fi; \
	done

tests: course-tests llm-tests learning-tests lab-tests

learning-tests:
	$(PY) -m unittest discover -s learning_projects/tests -v

lab-tests:
	node --test interactive/tests/engine.test.js

browser-tests:
	PY=$(PY) node interactive/tests/run-browser.js

lab:
	@echo "Visit http://localhost:8000/interactive/ — Ctrl+C stops the local server."
	$(PY) -m http.server 8000 --bind 127.0.0.1

course-tests:
	@for w in common $(WEEKS) scripts; do \
	  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. $(PY) -m pytest "$$w/tests" -q --maxfail=3 || exit 1; \
	done

qa: all
	$(PY) scripts/qa_report.py

# Optional lab: install llm-lab/requirements.txt first. No model downloads in tests.
llm-tests:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=llm-lab $(PY) -m pytest llm-lab/tests -q

llm-notebook:
	PYTHONPATH=llm-lab $(PY) -m llm_lab.execute_workbook

release-check:
	$(PY) scripts/check_release.py

secrets:
	gitleaks git --redact --log-opts="--all --full-history" .

clean:
	$(PY) scripts/clean_generated.py

cleanall: clean
