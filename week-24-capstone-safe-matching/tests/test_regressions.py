import numpy as np
import torch
from project.data import generate_dataset
from project.offline_iql import iql_pretrain, IQLConfig
from project.online_ppo_lag import ppo_lag_finetune, PPOLagConfig, expected_parity_gap
from project.audit import full_audit, demographic_parity_gap


def test_finetune_preserves_offline_policy_and_supports_excluded_group_feature():
    torch.set_num_threads(1)
    data = generate_dataset(n_candidates=30, n_jobs=2, n_interactions=30)
    base = iql_pretrain(data, IQLConfig(n_epochs=1, include_group=False))["model"]
    before = {key: value.clone() for key, value in base.state_dict().items()}
    output = ppo_lag_finetune(data, base, PPOLagConfig(n_updates=1, steps_per_update=32))
    assert output["net"].base is not base
    assert all(torch.equal(value, before[key]) for key, value in base.state_dict().items())
    audit = full_audit(output["net"], data, n_episodes=400)
    assert audit["n_episodes"] == 30
    assert audit["red_team"]["group_flip_disagreement"] == 0


def test_probability_gap_penalty_has_gradient_toward_parity():
    logits = torch.tensor([[0., 2.], [0., -2.]], requires_grad=True)
    gap = expected_parity_gap(logits, torch.tensor([0, 1]))
    gap.backward()
    assert logits.grad[0, 1] > 0  # gradient descent reduces over-selected group
    assert logits.grad[1, 1] < 0


def test_fairness_gap_is_unmeasured_when_a_group_is_absent():
    assert np.isnan(demographic_parity_gap(np.ones(3), np.zeros(3)))
