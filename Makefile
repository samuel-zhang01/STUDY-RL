# STUDY-RL — build orchestration

LATEXMK := latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error
WEEKS := $(sort $(wildcard week-*))
PY := python3

.PHONY: all syllabus weeks notebooks tests qa clean cleanall

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
	    papermill --kernel python3 $$w/workbook.ipynb $$w/workbook.executed.ipynb || exit 1; \
	  fi; \
	done

tests:
	pytest -q --maxfail=3

qa: all
	$(PY) scripts/qa_report.py

clean:
	@for w in $(WEEKS); do (cd $$w && latexmk -C) ; done
	cd syllabus && latexmk -C

cleanall: clean
	rm -rf $(foreach w,$(WEEKS),$(w)/_minted-*) syllabus/_minted-*
