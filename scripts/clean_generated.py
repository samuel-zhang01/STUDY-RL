#!/usr/bin/env python3
"""Remove only ignored build caches; never remove tracked files or PDFs."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", ".venv-llm", "venv", "runs", "data", "checkpoints"}
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
              ".ipynb_checkpoints", "_minted"}
AUX_SUFFIXES = {".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk",
                ".bbl", ".bcf", ".blg", ".nav", ".snm", ".vrb", ".lof", ".lot"}


def clean(root: Path) -> int:
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    removed = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if name in SKIP_DIRS or path.is_symlink():
                dirs.remove(name)
            elif name in CACHE_DIRS or name.startswith("_minted-"):
                dirs.remove(name)
                if any(item == relative or item.startswith(relative + "/") for item in tracked):
                    continue
                if subprocess.run(["git", "check-ignore", "-q", relative + "/"], cwd=root).returncode == 0:
                    shutil.rmtree(path)
                    removed += 1
        for name in files:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            generated = (path.suffix in AUX_SUFFIXES or name.endswith(
                (".synctex.gz", ".run.xml", ".executed.ipynb")))
            if generated and relative not in tracked and not path.is_symlink():
                if subprocess.run(["git", "check-ignore", "-q", relative], cwd=root).returncode == 0:
                    path.unlink()
                    removed += 1
    print(f"Removed {removed} ignored cache files/directories; shipped PDFs retained.")
    return removed


if __name__ == "__main__":
    clean(ROOT)
