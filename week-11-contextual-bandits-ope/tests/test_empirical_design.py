"""Check the scientific design and hand-derived moments, without plotting."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from project.make_figures import (fixed_reward_environment, empirical_error_moments,
                                  context_mc_reference, independent_replicate_seeds)
from project.env import SyntheticLeadBandit
from project.train import softmax_policy


def test_new_dataset_does_not_change_reward_model_or_target():
    first = fixed_reward_environment(model_seed=3,data_seed=101)
    second = fixed_reward_environment(model_seed=3,data_seed=202)
    same = fixed_reward_environment(model_seed=3,data_seed=101)
    np.testing.assert_array_equal(first.theta, second.theta)
    np.testing.assert_array_equal(first.theta, SyntheticLeadBandit(seed=3).theta)
    x1, x2 = first.sample_context(), second.sample_context()
    assert not np.array_equal(x1,x2)
    np.testing.assert_array_equal(x1,same.sample_context())
    pi1, pi2 = softmax_policy(first), softmax_policy(second)
    np.testing.assert_array_equal(pi1(x1),pi2(x1))
    np.testing.assert_array_equal(first.all_means(x1),second.all_means(x1))


def test_finite_error_decomposition_matches_hand_calculation():
    # Estimates .4,.6,.7 relative to .5 give errors -.1,.1,.2.
    # Mean error=1/15; squared mean error=1/225;
    # centered-error variance=7/450; mean squared error=1/50.
    result=empirical_error_moments([.4,.6,.7],.5)
    assert result['mean_error']==pytest.approx(1/15)
    assert result['squared_mean_error']==pytest.approx(1/225)
    assert result['variance_ddof0']==pytest.approx(7/450)
    assert result['mse']==pytest.approx(1/50)
    assert abs(result['identity_residual'])<1e-15
    with pytest.raises(ValueError):
        empirical_error_moments([.1,np.nan],.5)


def test_context_reference_integrates_rewards_without_consuming_log_rng():
    env = fixed_reward_environment(0,100)
    twin = fixed_reward_environment(0,100)
    # Every action's conditional mean is .5 when theta is zero, so the
    # integrated reference has mean .5 and zero Monte Carlo sampling SE.
    env.theta[:]=0
    reference=context_mc_reference(env,lambda x:np.ones(env.K)/env.K,n=20,seed=400)
    assert reference['estimate']==pytest.approx(.5)
    assert reference['standard_error']==pytest.approx(0,abs=1e-15)
    assert reference['n_contexts']==20 and reference['exact_truth'] is False
    np.testing.assert_array_equal(env.sample_context(),twin.sample_context())


def test_replicate_streams_are_explicit_distinct_and_repeatable():
    rows=independent_replicate_seeds(400,3)
    assert rows==independent_replicate_seeds(400,3)
    seeds=[r[name] for r in rows for name in ['data_seed','logger_seed','fit_seed']]
    assert len(seeds)==len(set(seeds))==9
