# STUDY-RL — build orchestration

LATEXMK := latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error
WEEKS := $(sort $(wildcard week-*))
PY := python3

.PHONY: all syllabus weeks notebooks tests llm-tests llm-notebook qa clean cleanall

all: syllabus weeks notebooks tests

syllabus:
	cd syllabus && $(LATEXMK) syllabus.tex && cp syllabus.pdf ../syllabus.pdf

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

tests:
	@for w in common $(WEEKS) scripts; do \
	  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. $(PY) -m pytest "$$w/tests" -q --maxfail=3 || exit 1; \
	done
	$(MAKE) llm-tests

qa: all
	$(PY) scripts/qa_report.py

# Optional lab: install llm-lab/requirements.txt first. No model downloads in tests.
llm-tests:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=llm-lab $(PY) -m pytest llm-lab/tests -q

llm-notebook:
	PYTHONPATH=llm-lab $(PY) -m llm_lab.execute_workbook

clean:
	@for w in $(WEEKS); do (cd $$w && latexmk -C) ; done
	cd syllabus && latexmk -C

cleanall: clean
	rm -rf $(foreach w,$(WEEKS),$(w)/_minted-*) syllabus/_minted-*
