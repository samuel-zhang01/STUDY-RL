"""Gymnasium env helpers — version-tolerant constructors."""
from __future__ import annotations
import gymnasium as gym


def make_cliffwalking():
    for vid in ("CliffWalking-v1", "CliffWalking-v0"):
        try:
            return gym.make(vid)
        except Exception:
            continue
    raise RuntimeError("No CliffWalking env available")


def make_taxi():
    for vid in ("Taxi-v3", "Taxi-v4"):
        try:
            return gym.make(vid)
        except Exception:
            continue
    raise RuntimeError("No Taxi env available")
