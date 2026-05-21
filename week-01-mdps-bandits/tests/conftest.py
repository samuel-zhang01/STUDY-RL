"""Pre-load the local ``project`` package under a unique name so test
collection of multiple weeks does not collide on the shared ``project``
module name."""
import importlib
import importlib.util
import os
import sys

WEEK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(WEEK)
sys.path.insert(0, REPO)
# Wipe any cached ``project`` from a sibling week and import this one fresh.
for mod in list(sys.modules):
    if mod == "project" or mod.startswith("project."):
        del sys.modules[mod]
sys.path.insert(0, WEEK)
import project  # noqa: F401, E402
