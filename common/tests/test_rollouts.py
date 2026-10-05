"""Boundary regressions independent of stochastic training performance."""
import numpy as np
import gymnasium as gym
from common.rollouts import generalized_advantage_estimate, sync_vector_env, transition_next_observations
from common.gridworld import GridWorld
from common.tabular import policy_iteration, policy_evaluation


def test_time_limit_bootstraps_but_does_not_cross_reset():
    adv, ret = generalized_advantage_estimate(
        [1., 100.], [2., 50.], [1., 0.], 0., gamma=.9, lam=1.,
        next_values=[10., 0.], terminals=[0., 0.],
    )
    np.testing.assert_allclose(adv, [8., 50.])
    np.testing.assert_allclose(ret, [10., 100.])
    _, terminal_ret = generalized_advantage_estimate([1.], [2.], [1.], 10.,
        gamma=.9, next_values=[10.], terminals=[1.])
    np.testing.assert_allclose(terminal_ret, [1.])


def test_same_step_autoreset_keeps_real_final_observation():
    vec = sync_vector_env([lambda: gym.make('Pendulum-v1', max_episode_steps=1)])
    try:
        vec.reset(seed=0)
        nxt, _, term, trunc, infos = vec.step(np.zeros((1, 1), dtype=np.float32))
        final = transition_next_observations(nxt, infos, term | trunc)
        assert trunc[0] and not term[0]
        assert not np.array_equal(final, nxt)
        # Every subsequent call advances an episode, with no reset-only fake transition.
        _, _, _, trunc2, _ = vec.step(np.zeros((1, 1), dtype=np.float32))
        assert trunc2[0]
    finally:
        vec.close()


def test_policy_iteration_budget_returns_its_policies_values():
    env = GridWorld(rows=3, cols=3, goals=((2, 2),))
    values, policy, _ = policy_iteration(env, gamma=.9, max_iter=1)
    np.testing.assert_allclose(values, policy_evaluation(env, policy, gamma=.9))
