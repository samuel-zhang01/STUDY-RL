import numpy as np
import torch
from project.train import _posterior_policy, _sample_chunks
from project.world_model import RSSMState


def test_recurrent_acting_uses_previous_executed_action():
    class Model:
        calls = []
        def initial(self, batch, device):
            return RSSMState(torch.zeros(batch, 1), torch.zeros(batch, 1))
        def observe_step(self, state, action, obs):
            self.calls.append((state.h.clone(), action.clone()))
            return (RSSMState(state.h + 1, state.s),)
    class Agent:
        n_actions = 2
        def actor(self, feature):
            return torch.distributions.Categorical(logits=torch.tensor([[0., 10.]]))
    model = Model()
    policy = _posterior_policy(model, Agent(), torch.device("cpu"), greedy=True)
    assert policy(np.zeros(1)) == policy(np.zeros(1)) == 1
    torch.testing.assert_close(model.calls[0][1], torch.zeros(1, 2))
    torch.testing.assert_close(model.calls[1][1], torch.tensor([[0., 1.]]))
    assert model.calls[1][0].item() == 1


def test_short_world_model_sequences_mask_padding():
    episode = dict(obs=np.zeros((2, 4), np.float32), acts=np.array([1]),
                   rews=np.array([1.]), conts=np.array([0.]))
    *_, mask = _sample_chunks([episode], 1, 4, np.random.default_rng(0), return_mask=True)
    assert mask.tolist() == [[1., 0., 0., 0.]]
