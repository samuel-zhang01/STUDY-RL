"""A tiny Decision Transformer for the synthetic GridWorld dataset.

Architecture follows Chen et al.\\ 2021 with a *single* transformer block
(2 heads, 64 dim, 1 layer) -- enough to overfit the toy domain and to
exhibit return-conditioned generation.

Sequence layout per timestep: (R_t, s_t, a_t) ordered (R, s, a), so token
stream is (R_0, s_0, a_0, R_1, s_1, a_1, ...). Causal mask is standard.
The model predicts actions only at the *action* positions.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim: int = 25, n_actions: int = 4,
                 max_timesteps: int = 64, context_len: int = 8,
                 hidden: int = 64, n_heads: int = 2, n_layers: int = 1,
                 dropout: float = 0.05) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.context_len = context_len
        self.hidden = hidden

        self.state_embed = nn.Linear(state_dim, hidden)
        self.action_embed = nn.Embedding(n_actions, hidden)
        self.return_embed = nn.Linear(1, hidden)
        self.time_embed = nn.Embedding(max_timesteps + 1, hidden)
        self.ln_in = nn.LayerNorm(hidden)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.action_head = nn.Linear(hidden, n_actions)

    def forward(self, returns: torch.Tensor, states: torch.Tensor,
                actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """returns: (B, K, 1) float; states: (B, K, state_dim); actions: (B, K) long;
        timesteps: (B, K) long. Returns logits over actions, shape (B, K, n_actions)."""
        B, K = states.shape[:2]
        tpos = self.time_embed(timesteps)
        rt = self.return_embed(returns) + tpos
        st = self.state_embed(states) + tpos
        at = self.action_embed(actions) + tpos

        # Interleave (R, s, a) into a sequence of length 3K.
        tokens = torch.stack([rt, st, at], dim=2).reshape(B, 3 * K, self.hidden)
        tokens = self.ln_in(tokens)

        # Causal mask.
        L = 3 * K
        mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool),
                          diagonal=1)
        h = self.transformer(tokens, mask=mask, is_causal=True)
        # Pick action positions: indices 1, 4, 7, ... (every 3rd, offset 1 -> s positions
        # are the predictor of next action). In DT, we predict action_t from the
        # token AT the state position (index 3t+1).
        h_s = h[:, 1::3, :]
        return self.action_head(h_s)
