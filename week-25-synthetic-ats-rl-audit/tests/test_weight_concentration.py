"""ESS reports weight concentration and refuses an undefined ratio."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from project.ope import effective_sample_size


def test_ess_is_scale_invariant_and_requires_observed_target_weight():
    slate = np.array([[1.], [2.]])
    log = [(np.array([1.]), 0, 1., .5, slate)] * 3
    assert effective_sample_size(log, lambda s: np.array([1., 0.])) == 3
    with pytest.raises(ValueError, match="undefined"):
        effective_sample_size(log, lambda s: np.array([0., 1.]))
