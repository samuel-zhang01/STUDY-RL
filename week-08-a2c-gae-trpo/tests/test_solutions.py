"""Unit tests for week-08 reference solutions: GAE, KL, conjugate gradient."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from project.solutions import (  # noqa: E402
    compute_gae,
    nstep_advantage,
    diagonal_gaussian_kl,
    conjugate_gradient,
    flat_params,
    set_flat_params,
)


def test_gae_lambda_zero_equals_td():
    rng = np.random.default_rng(0)
    T = 12
    r = rng.normal(size=T).astype(np.float64)
    v = rng.normal(size=T).astype(np.float64)
    dones = np.zeros(T)
    last_v = float(rng.normal())
    adv, _ = compute_gae(r, v, dones, last_v, gamma=0.9, lam=0.0)
    # lam=0 collapses to one-step TD: A_t = r_t + gamma * v_{t+1} - v_t (with last using bootstrap)
    expected = r.copy()
    for t in range(T):
        v_next = last_v if t == T - 1 else v[t + 1]
        expected[t] = r[t] + 0.9 * v_next - v[t]
    assert np.allclose(adv, expected, atol=1e-6)


def test_gae_lambda_one_equals_monte_carlo_minus_v():
    rng = np.random.default_rng(1)
    T = 10
    gamma = 0.95
    r = rng.normal(size=T).astype(np.float64)
    v = rng.normal(size=T).astype(np.float64)
    dones = np.zeros(T)
    last_v = float(rng.normal())
    adv, _ = compute_gae(r, v, dones, last_v, gamma=gamma, lam=1.0)
    # MC return-to-go from r (including bootstrap last_v after step T)
    G = np.zeros(T)
    running = last_v
    for t in range(T - 1, -1, -1):
        running = r[t] + gamma * running
        G[t] = running
    assert np.allclose(adv, G - v, atol=1e-5)


def test_nstep_collapses_to_gae_no_dones_for_small_n():
    rng = np.random.default_rng(2)
    T = 8
    gamma = 0.9
    r = rng.normal(size=T).astype(np.float64)
    v = rng.normal(size=T).astype(np.float64)
    last_v = 0.0
    n = 1
    adv1 = nstep_advantage(r, v, gamma, n, bootstrap=last_v)
    # 1-step advantage should match GAE(lam=0).
    g0, _ = compute_gae(r, v, np.zeros(T), last_v, gamma=gamma, lam=0.0)
    assert np.allclose(adv1, g0, atol=1e-6)


def test_diagonal_kl_self_is_zero():
    mu = torch.randn(4, 3)
    log_std = torch.randn(4, 3) * 0.1
    kl = diagonal_gaussian_kl(mu, log_std, mu.clone(), log_std.clone())
    assert abs(kl.item()) < 1e-6


def test_diagonal_kl_matches_closed_form_for_unit_shifts():
    mu1 = torch.zeros(1, 1); mu2 = torch.ones(1, 1)
    log_std = torch.zeros(1, 1)  # std=1
    kl = diagonal_gaussian_kl(mu1, log_std, mu2, log_std)
    # KL(N(0,1)||N(1,1)) = 0.5 * (1-0)^2 = 0.5
    assert abs(kl.item() - 0.5) < 1e-6


def test_conjugate_gradient_solves_spd_system():
    rng = np.random.default_rng(3)
    n = 8
    M = rng.normal(size=(n, n))
    A = M @ M.T + np.eye(n) * 1.0  # SPD
    b = rng.normal(size=n)
    x_cg = conjugate_gradient(lambda v: A @ v, b, n_iters=50)
    x_ref = np.linalg.solve(A, b)
    assert np.allclose(x_cg, x_ref, atol=1e-6)


def test_flat_param_roundtrip():
    p1 = torch.randn(3, 2, requires_grad=True)
    p2 = torch.randn(4, requires_grad=True)
    flat = flat_params([p1, p2])
    flat2 = flat + 0.5
    set_flat_params([p1, p2], flat2)
    assert torch.allclose(p1.detach().reshape(-1), flat2[:6])
    assert torch.allclose(p2.detach(), flat2[6:])
