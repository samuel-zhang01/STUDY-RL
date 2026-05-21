#!/usr/bin/env bash
# Re-compile every notes.tex / extended_reading.tex / syllabus.tex with the
# updated shared style. Run in parallel-ish to keep it fast.
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
LATEXMK="latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error"
fail=0

build() {
    local dir="$1" stem="$2"
    cd "$ROOT/$dir"
    rm -f "$stem.aux" "$stem.log" "$stem.toc" "$stem.out" "$stem.fls" "$stem.fdb_latexmk" 2>/dev/null
    if $LATEXMK "$stem.tex" >/dev/null 2>&1; then
        echo "OK    $dir/$stem.pdf"
    else
        echo "FAIL  $dir/$stem.tex" >&2
        return 1
    fi
}

for w in week-*; do
    build "$w" notes      || fail=$((fail+1))
    build "$w" extended_reading || fail=$((fail+1))
done
build syllabus syllabus || fail=$((fail+1))

echo
echo "Failures: $fail"
[ "$fail" -eq 0 ]
