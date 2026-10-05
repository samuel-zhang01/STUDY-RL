import pytest
from project.env_pendulum import PendulumDR
from project.shield import DFAShield, NoSafeActionError
from project.solutions import exercise_7


def test_upright_pendulum_is_unstable_under_gravity():
    env = PendulumDR(randomise=False)
    env.reset()
    env.theta = .01
    env.theta_dot = 0.
    env.step(0.)
    assert env.theta > .01 and env.theta_dot > 0


def test_shield_refuses_when_every_action_is_unsafe():
    with pytest.raises(NoSafeActionError):
        DFAShield(lambda state, action: True).filter(0, [0, 1, 2, 3])


def test_infeasible_cbf_does_not_return_an_unsafe_proposal():
    with pytest.raises(ValueError, match="infeasible"):
        exercise_7(1., h_val=-1., dh_da=0.)
