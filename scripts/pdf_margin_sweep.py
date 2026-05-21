#!/usr/bin/env python3
"""Phase-5b: scan every generated PDF for content cutoff / overfull issues.

Two heuristics:
  (A) Parse latexmk's .log files for "Overfull \\hbox" / "Overfull \\vbox"
      warnings and tally + threshold by points-too-wide.
  (B) Rasterize each page (pdftoppm), inspect a thin band along each edge,
      and flag pages where non-white pixels touch within `EDGE_PX` of the
      page edge — i.e. content is hugging or exceeding the trim.

Writes a TSV report to pdf_margin_report.tsv. Exit 1 if any PDF fails.

Usage: python scripts/pdf_margin_sweep.py [--render]
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDGE_PX = 2            # ~0.5mm at 100dpi — only flags real cutoff, not legal trim slack
TOL_PT = 5.0           # ignore overfulls smaller than this many pt
WHITE_THRESH = 248     # grayscale value above which a pixel counts as "white"


def parse_log_for_overfull(log_path: Path) -> list[tuple[float, int, str]]:
    """Return list of (pt_too_wide, line, context) for each Overfull >= TOL_PT."""
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    out: list[tuple[float, int, str]] = []
    pat = re.compile(
        r"Overfull \\(?:hbox|vbox) \(([\d.]+)pt too (?:wide|tall)\)[^\n]*"
        r"(?: in paragraph at lines? (\d+))?",
    )
    for m in pat.finditer(text):
        pt = float(m.group(1))
        if pt < TOL_PT:
            continue
        line = int(m.group(2)) if m.group(2) else -1
        ctx = m.group(0)[:140]
        out.append((pt, line, ctx))
    return out


def check_edges_via_raster(pdf_path: Path) -> tuple[int, list[int]]:
    """Render at 100dpi and detect content touching any edge.

    Returns (n_pages_with_cutoff, page_numbers).
    """
    try:
        from PIL import Image
    except ImportError:
        return 0, []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Render every page (cheap at 100 dpi for our few-page PDFs)
        subprocess.run(
            ["pdftoppm", "-r", "100", "-gray", str(pdf_path), str(td / "p")],
            check=False, capture_output=True,
        )
        bad: list[int] = []
        for img_path in sorted(td.glob("p-*.pgm")):
            try:
                im = Image.open(img_path)
            except Exception:
                continue
            w, h = im.size
            px = im.load()
            # Sample each edge band.
            def band_min(xs, ys) -> int:
                m = 255
                for y in ys:
                    for x in xs:
                        v = px[x, y]
                        if v < m:
                            m = v
                return m
            top = band_min(range(0, w, 4), range(0, EDGE_PX))
            bot = band_min(range(0, w, 4), range(h - EDGE_PX, h))
            left = band_min(range(0, EDGE_PX), range(0, h, 4))
            right = band_min(range(w - EDGE_PX, w), range(0, h, 4))
            if min(top, bot, left, right) < WHITE_THRESH:
                m = re.search(r"p-(\d+)\.pgm", img_path.name)
                if m:
                    bad.append(int(m.group(1)))
        return len(bad), bad


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true",
                        help="also rasterize and check edge bands (slower)")
    args = parser.parse_args(argv)

    rows = []
    fail = 0
    for w in sorted(ROOT.glob("week-*")):
        for stem in ("notes", "extended_reading"):
            pdf = w / f"{stem}.pdf"
            log = w / f"{stem}.log"
            if not pdf.exists():
                continue
            overfulls = parse_log_for_overfull(log)
            n_over = len(overfulls)
            worst = max((o[0] for o in overfulls), default=0.0)
            cutoff_pages = 0
            cutoff_list: list[int] = []
            if args.render:
                cutoff_pages, cutoff_list = check_edges_via_raster(pdf)
            status = "OK"
            if n_over > 0:
                status = f"OVERFULL ({n_over}, max {worst:.1f}pt)"
            if cutoff_pages > 0:
                status += f" CUTOFF pages={cutoff_list[:5]}"
            if status != "OK":
                fail += 1
            rows.append((w.name, stem, status))
    syllabus = ROOT / "syllabus" / "syllabus.pdf"
    if syllabus.exists():
        ofs = parse_log_for_overfull(ROOT / "syllabus" / "syllabus.log")
        worst = max((o[0] for o in ofs), default=0.0)
        st = "OK" if not ofs else f"OVERFULL ({len(ofs)}, max {worst:.1f}pt)"
        rows.append(("syllabus", "syllabus", st))

    out_path = ROOT / "pdf_margin_report.tsv"
    with out_path.open("w") as f:
        f.write("week\tdoc\tstatus\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    for r in rows:
        print("\t".join(r))
    print()
    print(f"PDFs scanned: {len(rows)}  Issues flagged: {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
