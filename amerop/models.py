"""Contract specification, Black-Scholes reference prices and risk-neutral path simulation."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class OptionSpec:
    """A vanilla option on a stock following geometric Brownian motion.

    S0 spot, K strike, r risk-free rate, sigma volatility, T maturity in years,
    q continuous dividend yield, kind "put" or "call".
    """
    S0: float = 36.0
    K: float = 40.0
    r: float = 0.06
    sigma: float = 0.20
    T: float = 1.0
    q: float = 0.0
    kind: str = "put"

    def payoff(self, S):
        S = np.asarray(S, dtype=float)
        if self.kind == "put":
            return np.maximum(self.K - S, 0.0)
        return np.maximum(S - self.K, 0.0)

    def with_(self, **changes) -> "OptionSpec":
        return replace(self, **changes)


def bs_vec(S, K, r, q, sigma, tau, kind="put"):
    """Vectorised Black-Scholes European price for time-to-maturity tau > 0."""
    S = np.asarray(S, dtype=float)
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / sq
    d2 = d1 - sq
    if kind == "put":
        return K * np.exp(-r * tau) * ndtr(-d2) - S * np.exp(-q * tau) * ndtr(-d1)
    return S * np.exp(-q * tau) * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2)


def bs_price(spec: OptionSpec) -> float:
    """European price of the same contract (a lower bound for the American price)."""
    if spec.T <= 0:
        return float(spec.payoff(spec.S0))
    return float(bs_vec(spec.S0, spec.K, spec.r, spec.q, spec.sigma, spec.T, spec.kind))


def simulate_gbm(spec: OptionSpec, n_paths: int, n_steps: int, rng: np.random.Generator,
                 antithetic: bool = True, moment_match: bool = False) -> np.ndarray:
    """Simulate risk-neutral GBM paths, shape (n_paths, n_steps + 1), column 0 = S0.

    dS = (r - q) S dt + sigma S dW, discretised exactly in log space.
    antithetic: pair every normal draw Z with -Z.
    moment_match: rescale each time step's draws to exact mean 0 and variance 1.
    """
    dt = spec.T / n_steps
    half = (n_paths + 1) // 2 if antithetic else n_paths
    Z = rng.standard_normal((half, n_steps))
    if antithetic:
        Z = np.concatenate([Z, -Z])[:n_paths]
    if moment_match:
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    drift = (spec.r - spec.q - 0.5 * spec.sigma ** 2) * dt
    log_paths = np.cumsum(drift + spec.sigma * np.sqrt(dt) * Z, axis=1)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = spec.S0
    S[:, 1:] = spec.S0 * np.exp(log_paths)
    return S
