"""Binomial-tree pricing of American (and Bermudan) options.

crr          Cox-Ross-Rubinstein (1979) tree with optional exercise boundary and Greeks.
bbsr         Binomial Black-Scholes with Richardson extrapolation (Broadie & Detemple 1996):
             the last step uses the Black-Scholes value, which removes the CRR odd/even
             oscillation, and 2*P(n) - P(n/2) cancels the leading error term. Used as the
             high-accuracy benchmark.
bermudan_tree  The same tree with exercise allowed only on a fixed set of dates, the exact
             counterpart of LSMC's discrete exercise grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import OptionSpec, bs_vec


@dataclass
class TreeResult:
    price: float
    delta: float = np.nan
    gamma: float = np.nan
    theta: float = np.nan
    boundary_t: np.ndarray = field(default_factory=lambda: np.array([]))
    boundary_S: np.ndarray = field(default_factory=lambda: np.array([]))


def crr(spec: OptionSpec, n: int, american: bool = True, exercise_every: int | None = None,
        smooth: bool = False, boundary: bool = False) -> TreeResult:
    """Price on an n-step CRR tree.

    exercise_every: allow early exercise only every k steps (Bermudan); None = every step.
    smooth: use the Black-Scholes value at the last step (BBS).
    boundary: record the critical stock price S*(t) at each exercise step.
    """
    dt = spec.T / n
    u = np.exp(spec.sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((spec.r - spec.q) * dt) - d) / (u - d)
    disc = np.exp(-spec.r * dt)
    if not 0 < p < 1:
        raise ValueError("Tree is not arbitrage-free: reduce dt or check inputs")

    def layer(i):  # stock prices at step i, highest first
        return spec.S0 * u ** (i - 2.0 * np.arange(i + 1))

    if smooth:
        S = layer(n - 1)
        V = bs_vec(S, spec.K, spec.r, spec.q, spec.sigma, dt, spec.kind)
        start = n - 2
        if american:
            V = np.maximum(V, spec.payoff(S))
    else:
        V = spec.payoff(layer(n))
        start = n - 1

    keep, bt, bs = {}, [], []
    for i in range(start, -1, -1):
        S = layer(i)
        V = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if american and (exercise_every is None or i % exercise_every == 0):
            ex = spec.payoff(S)
            mask = (ex > V + 1e-12 * spec.K) & (ex > 0)  # tolerance: ignore floating-point ties
            if mask.any():
                V = np.where(mask, ex, V)
                if boundary:
                    bt.append(i * dt)
                    bs.append(S[mask].max() if spec.kind == "put" else S[mask].min())
        if i in (1, 2):
            keep[i] = (S.copy(), V.copy())

    res = TreeResult(price=float(V[0]))
    if 1 in keep and 2 in keep:
        S1, V1 = keep[1]
        S2, V2 = keep[2]
        res.delta = float((V1[0] - V1[1]) / (S1[0] - S1[1]))
        up = (V2[0] - V2[1]) / (S2[0] - S2[1])
        dn = (V2[1] - V2[2]) / (S2[1] - S2[2])
        res.gamma = float((up - dn) / (0.5 * (S2[0] - S2[2])))
        res.theta = float((V2[1] - V[0]) / (2 * dt))
    if boundary:
        res.boundary_t, res.boundary_S = np.array(bt[::-1]), np.array(bs[::-1])
    return res


def bbsr(spec: OptionSpec, n: int = 2000) -> float:
    """Benchmark American price: 2 * BBS(n) - BBS(n/2)."""
    n = n + (n % 2)
    return 2 * crr(spec, n, smooth=True).price - crr(spec, n // 2, smooth=True).price


def bermudan_tree(spec: OptionSpec, n_exercise: int, steps_per_date: int = 100) -> float:
    """Bermudan price with exercise on n_exercise equally spaced dates (excluding t=0).

    Averages two tree sizes to damp the CRR oscillation.
    """
    vals = [crr(spec, n_exercise * k, exercise_every=k).price for k in (steps_per_date, steps_per_date + 1)]
    return float(np.mean(vals))
