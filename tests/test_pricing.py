import numpy as np
import pytest

from amerop import (LSMCConfig, OptionSpec, bbsr, bermudan_tree, bs_price, crr, evaluate_policy,
                    exercise_boundary, fqi, lsmc, lspi, simulate_gbm)
from amerop.basis import FAMILIES, design

BASE = OptionSpec()  # Longstaff-Schwartz: S0=36, K=40, r=6%, sigma=20%, T=1 put


def test_european_tree_converges_to_black_scholes():
    for kind in ("put", "call"):
        spec = BASE.with_(kind=kind)
        tree = np.mean([crr(spec, n, american=False).price for n in (2000, 2001)])
        assert tree == pytest.approx(bs_price(spec), abs=2e-3)


def test_put_call_parity_black_scholes():
    s = BASE.with_(q=0.02)
    c, p = bs_price(s.with_(kind="call")), bs_price(s)
    assert c - p == pytest.approx(s.S0 * np.exp(-s.q * s.T) - s.K * np.exp(-s.r * s.T), abs=1e-10)


def test_american_call_without_dividends_equals_european():
    s = BASE.with_(kind="call", S0=40.0)
    assert crr(s, 1000).price == pytest.approx(crr(s, 1000, american=False).price, abs=1e-10)


def test_price_ordering_european_bermudan_american():
    eur, berm, amer = bs_price(BASE), bermudan_tree(BASE, 50), bbsr(BASE, 2000)
    assert eur < berm < amer
    assert amer >= BASE.payoff(BASE.S0)


@pytest.mark.parametrize("S0,sigma,T,ref", [(36, 0.2, 1, 4.487), (40, 0.2, 1, 2.319), (44, 0.4, 2, 5.647)])
def test_benchmark_matches_published_american_values(S0, sigma, T, ref):
    # high-accuracy American put values for the Longstaff-Schwartz (2001) test cases
    assert bbsr(BASE.with_(S0=S0, sigma=sigma, T=T), 2000) == pytest.approx(ref, abs=2e-3)


def test_bbsr_agrees_with_large_crr_tree():
    big = np.mean([crr(BASE, n).price for n in (4000, 4001)])
    assert bbsr(BASE, 1000) == pytest.approx(big, abs=1e-3)


def test_tree_greeks_signs():
    r = crr(BASE, 800)
    assert -1 < r.delta < 0 and r.gamma > 0


def test_tree_boundary_is_below_strike_and_rises_to_it():
    r = crr(BASE, 1000, boundary=True)
    assert (r.boundary_S < BASE.K).all()
    assert r.boundary_S[-1] > r.boundary_S[0]


def test_simulation_is_risk_neutral():
    S = simulate_gbm(BASE, 200_000, 10, np.random.default_rng(0))
    assert S[:, -1].mean() == pytest.approx(BASE.S0 * np.exp(BASE.r * BASE.T), rel=2e-3)


def test_moment_matching_gives_exact_moments():
    S = simulate_gbm(BASE, 1000, 5, np.random.default_rng(0), antithetic=False, moment_match=True)
    z = np.diff(np.log(S), axis=1)
    assert np.allclose(z.std(axis=0), z.std(axis=0)[0])


@pytest.mark.parametrize("family", FAMILIES)
def test_basis_has_constant_first_column(family):
    x = np.linspace(0.6, 1.0, 50)
    X, scale = design(x, family, 4)
    assert X.shape == (50, 5)
    assert np.allclose(X[:, 0], X[0, 0])
    X2, _ = design(x, family, 4, scale)
    assert np.allclose(X, X2)


def test_lsmc_matches_bermudan_benchmark():
    berm = bermudan_tree(BASE, 50)
    r = lsmc(BASE, LSMCConfig(n_paths=50_000, seed=1))
    assert abs(r.price - berm) < 3 * r.stderr + 0.01


def test_out_of_sample_policy_is_below_benchmark():
    berm = bermudan_tree(BASE, 50)
    r = lsmc(BASE, LSMCConfig(n_paths=20_000, seed=2))
    oos = simulate_gbm(BASE, 100_000, 50, np.random.default_rng(99))
    v, se = evaluate_policy(BASE, r.continuation, oos)
    assert v < berm + 3 * se
    assert v > berm - 0.05


@pytest.mark.parametrize("reg", ["ridge", "mlp", "hgb"])
def test_alternative_regressors_give_reasonable_prices(reg):
    r = lsmc(BASE, LSMCConfig(n_paths=10_000, regressor=reg, seed=0))
    assert 4.3 < r.price < 4.7


def test_lsmc_boundary_close_to_tree():
    tree = crr(BASE, 1000, boundary=True)
    r = lsmc(BASE, LSMCConfig(n_paths=50_000))
    t, s = exercise_boundary(BASE, r.continuation, 50)
    mid = (t > 0.2) & (t < 0.8)
    tree_s = np.interp(t[mid], tree.boundary_t, tree.boundary_S)
    assert np.abs(s[mid] - tree_s).mean() < 0.5


@pytest.mark.parametrize("method", [lspi, fqi])
def test_rl_methods_learn_good_policies(method):
    paths = simulate_gbm(BASE, 30_000, 50, np.random.default_rng(5))
    r = method(BASE, paths)
    assert r.converged
    oos = simulate_gbm(BASE, 50_000, 50, np.random.default_rng(6))
    v, _ = evaluate_policy(BASE, r.continuation, oos)
    assert 4.35 < v < bermudan_tree(BASE, 50) + 0.03
