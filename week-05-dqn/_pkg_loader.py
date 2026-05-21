"""Load `project/` as the `w05_project` package (hyphenated dir workaround)."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE / "project"


def load() -> None:
    if "w05_project" in sys.modules:
        return
    pkg_init = PROJ / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location(
        "w05_project", pkg_init, submodule_search_locations=[str(PROJ)])
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["w05_project"] = pkg
    pkg_spec.loader.exec_module(pkg)
    for sub in ("agent", "train", "solutions"):
        spec = importlib.util.spec_from_file_location(
            f"w05_project.{sub}", PROJ / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"w05_project.{sub}"] = mod
        spec.loader.exec_module(mod)


load()

from w05_project import agent, train, solutions  # noqa: E402,F401
