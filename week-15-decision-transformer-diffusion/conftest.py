"""Per-week pytest conftest: prepend this week's dir to sys.path and evict any
sibling `project` modules so tests import the correct project package."""
import os
import sys

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Clear sibling-week project modules at collection start so each week's tests
# see its own `project` package.
def pytest_collectstart(collector):  # noqa: D401
    for m in [k for k in list(sys.modules) if k == "project" or k.startswith("project.")]:
        del sys.modules[m]
