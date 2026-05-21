"""A tiny Recurrent State-Space Model (RSSM) for CartPole.

This is a deliberately minimal Dreamer-style world model:
- deterministic GRU on hidden state h_t,
- stochastic latent s_t with diagonal Gaussian posterior and prior,
- decoder for next observation,
- reward head and discount (continue) head.

Losses follow the KL-regularised ELBO of Dreamer V1/V2:
    L = E_q[ -log p(o_{t+1}|h_{t+1}, s_{t+1}) - log p(r_t) - log p(c_t) ]
        + beta * KL( q(s_{t+1}|...) || p(s_{t+1}|h_{t+1}) ).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    h: torch.Tensor
    s: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(self.h.detach(), self.s.detach())


class RSSM(nn.Module):
    def __init__(self, obs_dim: int = 4, act_dim: int = 2,
                 stoch_dim: int = 16, det_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim; self.act_dim = act_dim
        self.stoch_dim = stoch_dim; self.det_dim = det_dim

        # Embed (s, a) into the GRU input.
        self.input_fc = nn.Sequential(nn.Linear(stoch_dim + act_dim, hidden), nn.ELU())
        self.gru = nn.GRUCell(hidden, det_dim)

        # Prior over next s given h.
        self.prior_head = nn.Sequential(nn.Linear(det_dim, hidden), nn.ELU(),
                                        nn.Linear(hidden, 2 * stoch_dim))
        # Posterior over s given h, o.
        self.post_head = nn.Sequential(nn.Linear(det_dim + obs_dim, hidden), nn.ELU(),
                                       nn.Linear(hidden, 2 * stoch_dim))

        # Decoder p(o | h, s).
        self.decoder = nn.Sequential(nn.Linear(det_dim + stoch_dim, hidden), nn.ELU(),
                                     nn.Linear(hidden, obs_dim))
        # Reward p(r | h, s).
        self.reward_head = nn.Sequential(nn.Linear(det_dim + stoch_dim, hidden), nn.ELU(),
                                         nn.Linear(hidden, 1))
        # Continue (discount) flag p(c | h, s).
        self.cont_head = nn.Sequential(nn.Linear(det_dim + stoch_dim, hidden), nn.ELU(),
                                       nn.Linear(hidden, 1))

    def initial(self, batch: int, device: torch.device) -> RSSMState:
        return RSSMState(
            h=torch.zeros(batch, self.det_dim, device=device),
            s=torch.zeros(batch, self.stoch_dim, device=device),
        )

    def _split(self, x):
        mu, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = log_std.exp()
        return mu, std

    def imagine_step(self, prev: RSSMState, action_onehot: torch.Tensor) -> tuple[RSSMState, torch.Tensor, torch.Tensor]:
        inp = self.input_fc(torch.cat([prev.s, action_onehot], dim=-1))
        h = self.gru(inp, prev.h)
        mu, std = self._split(self.prior_head(h))
        eps = torch.randn_like(mu)
        s = mu + std * eps
        feat = torch.cat([h, s], dim=-1)
        r = self.reward_head(feat).squeeze(-1)
        cont = torch.sigmoid(self.cont_head(feat)).squeeze(-1)
        return RSSMState(h, s), r, cont

    def observe_step(self, prev: RSSMState, action_onehot: torch.Tensor, obs: torch.Tensor):
        """Returns (post_state, prior_mu, prior_std, post_mu, post_std, recon, r, c)."""
        inp = self.input_fc(torch.cat([prev.s, action_onehot], dim=-1))
        h = self.gru(inp, prev.h)
        prior_mu, prior_std = self._split(self.prior_head(h))
        post_mu, post_std = self._split(self.post_head(torch.cat([h, obs], dim=-1)))
        eps = torch.randn_like(post_mu)
        s = post_mu + post_std * eps
        feat = torch.cat([h, s], dim=-1)
        recon = self.decoder(feat)
        r = self.reward_head(feat).squeeze(-1)
        c = torch.sigmoid(self.cont_head(feat)).squeeze(-1)
        return RSSMState(h, s), prior_mu, prior_std, post_mu, post_std, recon, r, c


def kl_gaussian(mu_q, std_q, mu_p, std_p):
    """KL(N(mu_q, std_q^2) || N(mu_p, std_p^2)) summed over last dim."""
    var_q = std_q ** 2; var_p = std_p ** 2
    return 0.5 * (
        (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
        + 2 * (torch.log(std_p + 1e-8) - torch.log(std_q + 1e-8))
        - 1.0
    ).sum(dim=-1)


# Alias for symmetry with code in train.py
RewardHead = nn.Module
