"""Make the optional lab importable in both root and lab-only pytest runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
