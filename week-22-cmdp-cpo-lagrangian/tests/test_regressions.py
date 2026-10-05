import numpy as np
from project.solutions import exercise_5


def test_empirical_cvar_uses_fractional_probability_at_the_cutoff():
    # Half of three equally likely outcomes is 1.5 outcomes: all of -4 and
    # half of 2, giving (-4 + 1)/1.5 = -2, not the mean of the first two.
    assert exercise_5(np.array([-4., 2., 10.]), .5) == -2.
