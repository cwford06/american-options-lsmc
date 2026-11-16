"""Least-Squares Monte Carlo (Longstaff & Schwartz 2001) with pluggable continuation-value
estimators.

Backward induction over the exercise dates: at each date, regress the discounted
realised future cash flow on functions of the current stock price (by default only for
in-the-money paths). Exercise where the immediate payoff beats the fitted continuation
value. The price is the average discounted cash flow under that policy.

Estimators ("regressor"):
  ols    least squares on a basis-function expansion (basis.py)
  ridge  L2-penalised least squares on the same basis (intercept unpenalised)
  mlp    a small neural network, warm-started from the previous date's fit
  hgb    gradient-boosted trees, constrained to be monotone in S (a put's continuation
         value falls as S rises; a call's rises)

The in-sample estimate reuses the paths the policy was fitted on, so it carries some
foresight. evaluate_policy() prices the fitted policy on independent paths. That value
is an unbiased estimate of a feasible policy, so it's a lower bound on the true price.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .basis import design
from .models import OptionSpec, bs_price, simulate_gbm


@dataclass
class LSMCConfig:
    n_paths: int = 100_000
    n_steps: int = 50             # exercise dates (equally spaced, excluding t=0)
    basis: str = "laguerre"
    degree: int = 3
    regressor: str = "ols"        # ols | ridge | mlp | hgb
    alpha: float = 1e-4           # ridge penalty, scaled by the number of regression points
    itm_only: bool = True
    antithetic: bool = True
    moment_match: bool = False
    control_variate: bool = False
    mlp_hidden: tuple = (32, 32)
    seed: int = 0


class BasisRegression:
    def __init__(self, family, degree, alpha=0.0):
        self.family, self.degree, self.alpha = family, degree, alpha

    def fit(self, x, y):
        X, self.scale = design(x, self.family, self.degree)
        if self.alpha > 0:
            pen = self.alpha * len(y) * np.eye(X.shape[1])
            pen[0, 0] = 0.0
            self.coef = np.linalg.solve(X.T @ X + pen, X.T @ y)
        else:
            self.coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, x):
        return design(x, self.family, self.degree, self.scale)[0] @ self.coef


class MLPRegression:
    """Two-layer ReLU network on standardised moneyness. Each date starts from the
    previous (later) date's weights, since neighbouring continuation functions are similar."""

    X_MID, X_SCALE = 0.85, 0.15

    def __init__(self, hidden=(32, 32), seed=0, prev=None):
        from sklearn.neural_network import MLPRegressor
        if prev is not None:
            self.net = copy.deepcopy(prev.net)
            self.net.set_params(warm_start=True, max_iter=75)
        else:
            self.net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=1e-4,
                                    learning_rate_init=1e-3, max_iter=150, batch_size=512, tol=1e-6,
                                    n_iter_no_change=15, random_state=seed)

    def _x(self, x):
        return ((x - self.X_MID) / self.X_SCALE)[:, None]

    def fit(self, x, y):
        self.ym, self.ys = float(y.mean()), float(y.std()) or 1.0
        self.net.fit(self._x(x), (y - self.ym) / self.ys)
        return self

    def predict(self, x):
        return self.net.predict(self._x(x)) * self.ys + self.ym


class HGBRegression:
    def __init__(self, kind="put", seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(max_iter=30, max_leaf_nodes=8, min_samples_leaf=300,
                                                   learning_rate=0.15, random_state=seed,
                                                   monotonic_cst=[-1 if kind == "put" else 1])

    def fit(self, x, y):
        self.model.fit(x[:, None], y)
        return self

    def predict(self, x):
        return self.model.predict(x[:, None])


def _make(cfg: LSMCConfig, spec: OptionSpec, prev=None):
    if cfg.regressor == "ols":
        return BasisRegression(cfg.basis, cfg.degree)
    if cfg.regressor == "ridge":
        return BasisRegression(cfg.basis, cfg.degree, cfg.alpha)
    if cfg.regressor == "mlp":
        return MLPRegression(cfg.mlp_hidden, cfg.seed, prev)
    if cfg.regressor == "hgb":
        return HGBRegression(spec.kind, cfg.seed)
    raise ValueError(f"unknown regressor {cfg.regressor!r}")


@dataclass
class LSMCResult:
    price: float
    stderr: float
    models: list
    spec: OptionSpec
    cfg: LSMCConfig
    elapsed: float
    exercise_frac: float = 0.0
    tau: np.ndarray = field(default=None, repr=False)

    def continuation(self, t: int, x: np.ndarray) -> np.ndarray:
        m = self.models[t]
        return np.full(len(x), np.inf) if m is None else m.predict(x)


def lsmc(spec: OptionSpec, cfg: LSMCConfig = LSMCConfig(), paths: np.ndarray | None = None) -> LSMCResult:
    t0 = time.perf_counter()
    if paths is None:
        paths = simulate_gbm(spec, cfg.n_paths, cfg.n_steps, np.random.default_rng(cfg.seed),
                             cfg.antithetic, cfg.moment_match)
    S = paths
    N, M = S.shape[0], S.shape[1] - 1
    disc = np.exp(-spec.r * spec.T / M)
    cf = spec.payoff(S[:, M])
    tau = np.full(N, M)
    models = [None] * (M + 1)
    prev = None
    min_pts = max(cfg.degree + 2, 20)
    for t in range(M - 1, 0, -1):
        ex = spec.payoff(S[:, t])
        idx = np.flatnonzero(ex > 0) if cfg.itm_only else np.arange(N)
        if len(idx) < min_pts:
            continue
        x = S[idx, t] / spec.K
        y = cf[idx] * disc ** (tau[idx] - t)
        reg = _make(cfg, spec, prev).fit(x, y)
        cont = reg.predict(x)
        stop = idx[(ex[idx] > cont) & (ex[idx] > 0)]
        cf[stop] = ex[stop]
        tau[stop] = t
        models[t] = prev = reg

    pv = cf * disc ** tau
    if cfg.control_variate:
        # European put on the same paths, whose exact mean is the Black-Scholes price.
        # With antithetic paths the coefficient is estimated on pair averages, the units
        # the estimator actually averages over.
        ye = np.exp(-spec.r * spec.T) * spec.payoff(S[:, M])
        if cfg.antithetic and N % 2 == 0:
            h = N // 2
            c = np.cov(0.5 * (pv[:h] + pv[h:]), 0.5 * (ye[:h] + ye[h:]))
        else:
            c = np.cov(pv, ye)
        beta = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0
        pv = pv - beta * (ye - bs_price(spec))
    price = max(float(pv.mean()), float(spec.payoff(spec.S0)))
    return LSMCResult(price=price, stderr=_stderr(pv, cfg.antithetic), models=models, spec=spec,
                      cfg=cfg, elapsed=time.perf_counter() - t0, exercise_frac=float((tau < M).mean()), tau=tau)


def _stderr(pv: np.ndarray, antithetic: bool) -> float:
    """Standard error of the mean. Antithetic pairs (path i and i + N/2) are not independent,
    so they are averaged first; treating them as independent would hide their benefit."""
    n = len(pv)
    if antithetic and n % 2 == 0:
        pairs = 0.5 * (pv[: n // 2] + pv[n // 2:])
        return float(pairs.std(ddof=1) / np.sqrt(n // 2))
    return float(pv.std(ddof=1) / np.sqrt(n))


def evaluate_policy(spec: OptionSpec, continuation: Callable[[int, np.ndarray], np.ndarray],
                    paths: np.ndarray) -> tuple[float, float]:
    """Value of the exercise policy 'stop when payoff > continuation(t, S/K)' on new paths."""
    S = paths
    N, M = S.shape[0], S.shape[1] - 1
    disc = np.exp(-spec.r * spec.T / M)
    alive = np.ones(N, dtype=bool)
    pv = np.zeros(N)
    for t in range(1, M):
        ex = spec.payoff(S[:, t])
        cand = np.flatnonzero(alive & (ex > 0))
        if len(cand) == 0:
            continue
        stop = cand[ex[cand] > continuation(t, S[cand, t] / spec.K)]
        pv[stop] = ex[stop] * disc ** t
        alive[stop] = False
    pv[alive] = spec.payoff(S[alive, M]) * disc ** M
    value = max(float(pv.mean()), float(spec.payoff(spec.S0)))
    return value, float(pv.std(ddof=1) / np.sqrt(N))


def exercise_boundary(spec: OptionSpec, continuation: Callable[[int, np.ndarray], np.ndarray],
                      n_steps: int, grid_size: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Critical price S*(t) implied by a continuation-value estimate, found on a price grid.

    For a put, S* is the highest price at which exercising beats continuing.
    """
    dt = spec.T / n_steps
    if spec.kind == "put":
        grid = np.linspace(0.3 * spec.K, spec.K, grid_size, endpoint=False)
    else:
        grid = np.linspace(spec.K, 3.0 * spec.K, grid_size)[1:]
    ts, bs = [], []
    for t in range(1, n_steps):
        cont = continuation(t, grid / spec.K)
        stop = spec.payoff(grid) > cont
        if np.isfinite(cont).all() and stop.any():
            ts.append(t * dt)
            bs.append(grid[stop].max() if spec.kind == "put" else grid[stop].min())
    return np.array(ts), np.array(bs)
