"""Pre-load the local ``project`` package under a unique name so test
collection of multiple weeks does not collide on the shared ``project``
module name."""
import os
import sys

WEEK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(WEEK)
sys.path.insert(0, REPO)
for mod in list(sys.modules):
    if mod == "project" or mod.startswith("project."):
        del sys.modules[mod]
sys.path.insert(0, WEEK)
import project  # noqa: F401, E402
