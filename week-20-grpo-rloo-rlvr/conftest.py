"""Make ``project`` importable as a top-level package for this week's tests.

Because the week folder name has hyphens, it isn't a Python identifier; we
side-step that by inserting the week folder onto ``sys.path``, so test files
can ``import project`` directly.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
