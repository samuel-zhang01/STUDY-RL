"""Regressions for authoring and artifact QA; no training or PDF compiler."""
import json
import subprocess

import pytest

from scripts import new_week, pdf_margin_sweep, qa_report


def test_new_week_validates_paths_pads_number_and_escapes_tex(tmp_path, monkeypatch):
    monkeypatch.setattr(new_week, "ROOT", tmp_path)
    assert new_week.main("7", "../escape", "Title") == 2
    assert not list(tmp_path.iterdir())
    assert new_week.main("7", "test-topic", "A & B_50%") == 0
    folder = tmp_path / "week-07-test-topic"
    assert r"A \& B\_50\%" in (folder / "notes.tex").read_text()
    assert "Empirical results" in (folder / "notes.tex").read_text()
    notebook = json.loads((folder / "workbook.ipynb").read_text())
    assert "A & B_50%" in "".join(notebook["cells"][0]["source"])


def test_inventory_fails_for_missing_bundle_components_not_executed_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(qa_report, "ROOT", tmp_path)
    week = tmp_path / "week-99-test"
    week.mkdir()
    for filename in ("notes.pdf", "notes.tex", "workbook.ipynb", "teaching_guide.md",
                     "extended_reading.tex", "extended_reading.pdf"):
        (week / filename).touch()
    (week / "tests").mkdir()
    assert qa_report.main() == 1
    (week / "project").mkdir()
    (week / "project" / "solutions.py").touch()
    assert qa_report.main() == 0  # inventory does not claim fresh execution


def test_overfull_report_preserves_tex_line(tmp_path):
    log = tmp_path / "notes.log"
    log.write_text(r"Overfull \hbox (8.5pt too wide) in paragraph at lines 42--44" + "\n")
    assert pdf_margin_sweep.parse_log_for_overfull(log)[0][:2] == (8.5, 42)


def test_failed_rasterization_cannot_report_no_cutoff(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    def fail(*args, **kwargs):
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(pdf_margin_sweep.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        pdf_margin_sweep.check_edges_via_raster(tmp_path / "bad.pdf")
