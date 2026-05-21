import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np
import torch
from project import solutions as S


def test_kl():
    mu = torch.zeros(4, 8); std = torch.ones(4, 8)
    k = S.exercise_1_kl_diag_gaussian(mu, std, mu, std)
    assert torch.allclose(k, torch.zeros(4), atol=1e-5)


def test_lambda_return_constant_rewards():
    H, B = 4, 2
    rs = [torch.ones(B) for _ in range(H)]
    vs = [torch.zeros(B) for _ in range(H + 1)]
    cs = [torch.ones(B) for _ in range(H)]
    G = S.exercise_2_lambda_return(rs, vs, cs, gamma=0.99, lam=1.0)
    # With lam=1 and v=0, G_t == sum_{k>=t} gamma^{k-t} = (1 - gamma^(H-t)) / (1 - gamma)
    expected_first = (1 - 0.99 ** H) / (1 - 0.99)
    assert abs(float(G[0, 0]) - expected_first) < 1e-3


def test_rssm_imagine():
    out = S.exercise_3_rssm_imagine(seed=0)
    assert np.isfinite(out["feat_norm"])
    assert out["r"].shape == (2,)


def test_kl_balanced_nonzero_grad():
    mu_q = torch.zeros(3, requires_grad=True)
    std_q = torch.ones(3, requires_grad=True)
    mu_p = torch.ones(3, requires_grad=True)
    std_p = torch.ones(3, requires_grad=True)
    loss = S.exercise_5_kl_balanced(mu_q, std_q, mu_p, std_p, alpha=0.8).sum()
    loss.backward()
    assert mu_q.grad is not None
