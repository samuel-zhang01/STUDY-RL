"""Week 21 --- reward hacking and scalable oversight lab.

Two mini-pipelines:

- :mod:`hack_lab` : a deliberately gameable reward and three mitigations
  (KL anchor, length penalty, entropy bonus). Trains a tiny char-level LM
  and visualises hack-vs-true-quality.
- :mod:`cai` : a Constitutional AI mini-loop. SFT model generates --> rule
  critique --> revise --> preference pair --> DPO step.
"""
from . import hack_lab, cai
from . import solutions

__all__ = ["hack_lab", "cai", "solutions"]
