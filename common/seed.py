"""Reproducibility utilities."""
from __future__ import annotations
import os
import random
import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs.

    PYTHONHASHSEED affects child processes; this process's hash seed was
    chosen at interpreter startup. CUDA kernels are not made deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # speed > strict determinism
    except ImportError:
        pass
