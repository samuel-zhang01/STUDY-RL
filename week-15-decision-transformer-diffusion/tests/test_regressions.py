import numpy as np
import torch
from project.data import build_dt_dataset
from project.model import DecisionTransformer


def test_dataset_preserves_every_episode_and_resets_timestep():
    episodes, _, _, _ = build_dt_dataset(n_episodes=10, seed=3)
    assert len(episodes) == 10
    assert all(ep["timesteps"][0] == 0 for ep in episodes)


def test_masked_padding_cannot_change_valid_action_prediction():
    torch.manual_seed(0)
    model = DecisionTransformer(state_dim=2, n_actions=2, hidden=8, dropout=0).eval()
    states = torch.tensor([[[0., 0.], [1., 0.]]])
    returns = torch.zeros(1, 2, 1)
    actions = torch.zeros(1, 2, dtype=torch.long)
    times = torch.zeros_like(actions)
    mask = torch.tensor([[False, True]])
    before = model(returns, states, actions, times, attention_mask=mask)[:, -1]
    states[:, 0] = 100.
    returns[:, 0] = -100.
    actions[:, 0] = 1
    after = model(returns, states, actions, times, attention_mask=mask)[:, -1]
    torch.testing.assert_close(before, after)
