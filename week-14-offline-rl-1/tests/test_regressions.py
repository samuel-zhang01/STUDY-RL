import numpy as np
from project.agent import IQLAgent, NaiveQAgent
from project.env import collect_logged_dataset
from project.solutions import exercise_7_off_policy_evaluation_is


def test_time_limits_end_each_logged_episode_without_terminal_bootstrap_mask():
    data = collect_logged_dataset(n_episodes=3, max_steps=1)
    assert data["episode_end"].tolist() == [True, True, True]
    assert data["done"].tolist() == [0.0, 0.0, 0.0]


def test_trajectory_snips_resets_at_collection_limits():
    data = dict(done=np.zeros(2), episode_end=np.ones(2, bool), s_idx=np.zeros(2, int),
                a=np.zeros(2, int), r=np.array([1., 3.]),
                a_log_prob=np.log([.5, .25]), nA=2)
    assert np.isclose(exercise_7_off_policy_evaluation_is(data, np.array([0])), 7 / 3)


def test_awr_respects_logger_frequencies_and_excludes_unsupported_actions():
    agent = IQLAgent(nS=1, nA=3)
    agent.fit_behavior(np.zeros(10, int), np.array([0] * 3 + [1] * 7))
    agent.Q[0, 2] = 100.  # optimistic but absent in the dataset
    np.testing.assert_allclose(agent.policy_probs(), [[.3, .7, 0.]], atol=1e-6)
    assert agent.policy().tolist() == [1]


def test_repeating_one_transition_does_not_multiply_learning_rate():
    agent = NaiveQAgent(nS=1, nA=1, lr_q=.1)
    agent.update(np.zeros(64, int), np.zeros(64, int), np.ones(64),
                 np.zeros(64, int), np.ones(64))
    assert np.isclose(agent.Q[0, 0], .1)
