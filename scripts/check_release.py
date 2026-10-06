#!/usr/bin/env python3
"""Check the working release tree without executing notebooks or model code.

Use Gitleaks separately for credential and history scanning. This check validates
source hygiene and local links, not research claims or production safety.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
FORBIDDEN_DIRS = {".git", ".claude", ".codex", ".venv", ".venv-llm",
                  "__pycache__", ".ipynb_checkpoints", "runs", "checkpoints"}
FORBIDDEN_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".pt", ".pth",
                      ".ckpt", ".safetensors", ".bundle"}
PERSONAL_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@(?:gmail|hotmail|icloud|outlook|yahoo)\.com\b", re.I)
LOCAL_PATH = re.compile(r"/(?:Users|home)/[^\s\"'<>]+")


def source_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True)
    return sorted({root / name.decode() for name in result.stdout.split(b"\0")
                   if name and (root / name.decode()).is_file()})


def check_links(text: str, path: Path, root: Path) -> list[str]:
    # Example snippets are not live documentation links.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    errors = []
    for match in LINK.finditer(text):
        target = unquote(match[1].strip("<>"))
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        resolved = (path.parent / url.path).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            errors.append(f"{path.relative_to(root)}: broken local link {target}")
    return errors


def check_path(path: Path, root: Path) -> list[str]:
    relative = path.relative_to(root)
    errors = []
    if (FORBIDDEN_DIRS.intersection(relative.parts)
            or path.suffix in FORBIDDEN_SUFFIXES
            or path.name.endswith((".executed.ipynb", ".local.json"))
            or (path.name.startswith(".env") and path.name != ".env.example")
            or path.name in {"credentials.json", "id_rsa", "id_ed25519"}):
        errors.append(f"{relative}: local/private artifact included in release")
    if path.is_symlink() and not path.resolve().is_relative_to(root.resolve()):
        errors.append(f"{relative}: symlink points outside the repository")
    if path.suffix not in {".py", ".md", ".ipynb", ".tex", ".json", ".jsonl", ".sh", ".tsv"}:
        return errors
    text = path.read_text(encoding="utf-8")
    if PERSONAL_EMAIL.search(text) or LOCAL_PATH.search(text):
        errors.append(f"{relative}: personal email or machine-specific home path")
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=str(relative))
        except SyntaxError as exc:
            errors.append(f"{relative}:{exc.lineno}: {exc.msg}")
    elif path.suffix == ".md":
        errors.extend(check_links(text, path, root))
    elif path.suffix == ".ipynb":
        import nbformat
        try:
            notebook = nbformat.reads(text, as_version=4)
            nbformat.validate(notebook)
        except (ValueError, nbformat.ValidationError) as exc:
            errors.append(f"{relative}: invalid notebook: {exc}")
            return errors
        for index, cell in enumerate(notebook.cells, 1):
            if cell.cell_type == "markdown":
                errors.extend(check_links(cell.source, path, root))
            elif cell.cell_type == "code":
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    errors.append(f"{relative}: cell {index} contains saved execution")
                try:
                    ast.parse(cell.source)
                except SyntaxError as exc:
                    errors.append(f"{relative}: cell {index}: {exc.msg}")
    elif path.suffix == ".json":
        json.loads(text)
    elif path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                json.loads(line)
    return errors


def main() -> int:
    errors = []
    for name in ("README.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md",
                 "THIRD_PARTY_NOTICES.md", "docs/RELEASE_REVIEW.md"):
        if not (ROOT / name).is_file():
            errors.append(f"Missing release document: {name}")
    paths = source_paths(ROOT)
    for path in paths:
        try:
            errors.extend(check_path(path, ROOT))
        except (ValueError, OSError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for error in errors:
        print(error)
    print(f"Release tree: {len(paths)} files checked; {len(errors)} issue(s).")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
