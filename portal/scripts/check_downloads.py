#!/usr/bin/env python3
"""Validate hosted bundles against canonical sources and import them in isolation.

Run after prepare_content.py. Setup cells only: no training or model downloads.
The active Python environment must contain the course's CPU dependencies.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "portal/generated/assets"


def main():
    count = 0
    for week in sorted(ROOT.glob("week-*")):
        notebook = week / "workbook.ipynb"
        if not notebook.is_file():
            continue
        bundle = ASSETS / f"bundles/week-{week.name.split('-')[1]}.zip"
        with zipfile.ZipFile(bundle) as archive, tempfile.TemporaryDirectory(prefix="rl-download-") as temp:
            names = archive.namelist()
            assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
            for required in [f"{week.name}/workbook.ipynb", "common/seed.py", "conftest.py",
                             "requirements.txt", "requirements-foundations.txt", "LICENSE",
                             f"{week.name}/notes.pdf", f"{week.name}/extended_reading.pdf"]:
                assert required in names, f"{bundle.name}: missing {required}"
                assert archive.read(required) == (ROOT / required).read_bytes(), f"Stale {required}"
            assert not any("__pycache__" in n or ".executed." in n or ".env" in Path(n).parts for n in names)
            archive.extractall(temp)
            cells = json.loads(notebook.read_text())["cells"]
            setup = "".join(next(c for c in cells if c["cell_type"] == "code")["source"])
            env = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            env.pop("PYTHONPATH", None)
            run = subprocess.run([sys.executable, "-c", setup],
                                 cwd=Path(temp) / week.name, env=env, capture_output=True, text=True, timeout=60)
            if run.returncode:
                raise RuntimeError(f"{week.name}: extracted notebook setup failed:\n{run.stderr[-1800:]}")
        count += 1
    assert count == 26, f"Expected 26 bundles, found {count}"
    print(f"PASS: {count} complete bundles match source and all extracted notebook setup cells import.")


if __name__ == "__main__":
    main()
