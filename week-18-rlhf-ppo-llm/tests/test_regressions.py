import copy
import pytest
import torch
from project.model import CharTransformer, TransformerConfig, ValueHead, RewardModel
from project.ppo import sample_rollouts, ppo_step


def test_unchanged_policy_has_unit_ratio_at_nonunit_sampling_temperature():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = CharTransformer(TransformerConfig(d_model=8, n_layer=1))
    ref = copy.deepcopy(policy)
    head = ValueHead(8)
    rm = RewardModel(copy.deepcopy(policy))
    rollout = sample_rollouts(policy, head, ref, rm, ["Q: tea? A:"], 2, temperature=.7)
    opt = torch.optim.SGD(list(policy.parameters()) + list(head.parameters()), lr=0)
    diagnostics = ppo_step(policy, head, rollout, opt, n_epochs=1)
    assert diagnostics["approx_kl"] == pytest.approx(0, abs=1e-5)
    assert diagnostics["reference_kl"] == pytest.approx(0, abs=1e-5)


def test_reference_drift_diagnostic_is_distinct_from_ppo_update_kl():
    torch.manual_seed(2)
    policy = CharTransformer(TransformerConfig(d_model=8, n_layer=1))
    ref = copy.deepcopy(policy)
    head = ValueHead(8)
    rollout = sample_rollouts(policy, head, ref, RewardModel(copy.deepcopy(policy)), ["A:"], 2)
    rollout.ref_logprobs = rollout.logprobs - .5
    opt = torch.optim.SGD(list(policy.parameters()) + list(head.parameters()), lr=0)
    diagnostics = ppo_step(policy, head, rollout, opt, n_epochs=1)
    assert diagnostics["reference_kl"] == pytest.approx(.5)
    assert diagnostics["approx_kl"] == pytest.approx(0, abs=1e-5)
