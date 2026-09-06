#!/usr/bin/env bash
# Inventory existing artifacts and run each week's tests in a fresh process.
# This script does not compile PDFs or execute notebooks; use make all for those.
# Writes a TSV report to qa_report.tsv. Exits 0 if all green.
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
REPORT="qa_report.tsv"
echo -e "week\tnotes.pdf\textended_reading.pdf\tworkbook.ipynb\tworkbook.executed\ttests\tnotes_pages" > "$REPORT"

pass=0; fail=0
for w in week-*/; do
    w="${w%/}"
    cd "$ROOT/$w"
    notes_pdf="-"
    ext_pdf="-"
    wb_present="-"
    wb_exec="-"
    tests="-"
    pages="-"

    [ -f notes.pdf ] && notes_pdf="OK" && pages="$(pdfinfo notes.pdf 2>/dev/null | awk '/^Pages:/{print $2}')" || notes_pdf="MISS"
    [ -f extended_reading.pdf ] && ext_pdf="OK" || ext_pdf="MISS"
    [ -f workbook.ipynb ] && wb_present="OK" || wb_present="MISS"
    [ -f workbook.executed.ipynb ] && wb_exec="OK" || wb_exec="-"

    if [ -d tests ]; then
        cd "$ROOT"
        test_log="$(mktemp)"
        if PYTHONPATH=. python3 -m pytest "$w/tests" -q --no-header --timeout=300 > "$test_log" 2>&1; then
            tests="OK"; pass=$((pass+1))
        else
            tests="FAIL"; fail=$((fail+1))
        fi
        tail -1 "$test_log"
        rm -f "$test_log"
        cd "$ROOT/$w"
    fi
    echo -e "$w\t$notes_pdf\t$ext_pdf\t$wb_present\t$wb_exec\t$tests\t$pages" >> "$ROOT/$REPORT"
done
cd "$ROOT"
column -ts $'\t' "$REPORT"
echo
echo "Pass: $pass  Fail: $fail"
[ "$fail" -eq 0 ]
