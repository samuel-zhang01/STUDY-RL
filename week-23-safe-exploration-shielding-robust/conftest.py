"""Expose this week's project package in an isolated pytest invocation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
