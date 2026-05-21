"""Load `project/` as the `w04_project` package (hyphenated dir workaround)."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE / "project"


def load() -> None:
    if "w04_project" in sys.modules:
        return
    pkg_init = PROJ / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location(
        "w04_project", pkg_init, submodule_search_locations=[str(PROJ)])
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["w04_project"] = pkg
    pkg_spec.loader.exec_module(pkg)
    for sub in ("agents", "env_utils", "train", "solutions"):
        spec = importlib.util.spec_from_file_location(
            f"w04_project.{sub}", PROJ / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"w04_project.{sub}"] = mod
        spec.loader.exec_module(mod)


load()

from w04_project import agents, env_utils, train, solutions  # noqa: E402,F401
