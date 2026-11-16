"""Reinforcement learning for optimal exercise: LSPI and Fitted Q-Iteration.

Exercising an American option is an optimal-stopping Markov decision process:

  state   s = (t, S_t)
  actions stop (receive payoff g(S_t), episode ends) or continue
  Q(s, stop) = g(S_t)        Q(s, continue) ~ w . phi(s)

Unlike LSMC, which fits a separate regression at each date by backward induction, both
methods here learn a single continuation function over (time, price) from all one-step
transitions (s, s') in the simulated paths.

LSPI, Least-Squares Policy Iteration (Lagoudakis & Parr 2003), applied to American options
by Li, Szepesvari & Schuurmans (2009). It alternates:
  policy evaluation (LSTD-Q): solve A w = b with
      A = sum phi(s) [phi(s) - gamma * 1{continue at s'} phi(s')]^T
      b = sum phi(s) gamma * 1{stop at s'} g(s')
  policy improvement: at s', stop iff g(s') > 0 and g(s') >= w . phi(s')
until w stops changing.

FQI, Fitted Q-Iteration (Ernst, Geurts & Wehenkel 2005; Tsitsiklis & Van Roy 2001):
  w <- argmin sum (w . phi(s) - gamma * max(g(s'), w_old . phi(s')))^2

Features: a constant, weighted Laguerre polynomials in S/K, the three time features from
Li et al. (sin(pi/2 (1 - t/T)), ln(1 - t/T), (t/T)^2) and, by default, their products
with the price features, so the shape in S can change over time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import laguerre

from .models import OptionSpec


def features(x: np.ndarray, t_frac: float, degree: int = 3, interactions: bool = True) -> np.ndarray:
    n = len(x)
    price = np.exp(-x / 2)[:, None] * laguerre.lagvander(x, degree - 1)
    tf = np.array([np.sin(np.pi / 2 * (1 - t_frac)), np.log(max(1 - t_frac, 1e-6)), t_frac ** 2])
    cols = [np.ones((n, 1)), price, np.broadcast_to(tf, (n, 3))]
    if interactions:
        cols += [price * tf[i] for i in range(3)]
    return np.hstack(cols)


@dataclass
class RLResult:
    method: str
    w: np.ndarray
    iterations: int
    converged: bool
    in_sample: float
    elapsed: float
    history: list
    n_steps: int
    degree: int
    interactions: bool

    def continuation(self, t: int, x: np.ndarray) -> np.ndarray:
        return features(x, t / self.n_steps, self.degree, self.interactions) @ self.w


def _setup(spec, paths, degree, interactions):
    S = paths
    M = S.shape[1] - 1
    phis = [features(S[:, t] / spec.K, t / M, degree, interactions) for t in range(M)]
    pays = [spec.payoff(S[:, t]) for t in range(M + 1)]
    return M, np.exp(-spec.r * spec.T / M), phis, pays


def _finish(method, spec, paths, w, it, converged, history, M, degree, interactions, t0):
    from .longstaff_schwartz import evaluate_policy
    res = RLResult(method, w, it, converged, np.nan, 0.0, history, M, degree, interactions)
    res.in_sample = evaluate_policy(spec, res.continuation, paths)[0]
    res.elapsed = time.perf_counter() - t0
    return res


def lspi(spec: OptionSpec, paths: np.ndarray, degree: int = 3, interactions: bool = True,
         max_iter: int = 30, tol: float = 1e-6, ridge: float = 1e-8) -> RLResult:
    t0 = time.perf_counter()
    N = paths.shape[0]
    M, gamma, phis, pays = _setup(spec, paths, degree, interactions)
    k = phis[0].shape[1]
    w, history, converged = np.zeros(k), [], False
    for it in range(1, max_iter + 1):
        A, b = ridge * N * np.eye(k), np.zeros(k)
        for t in range(M):
            ph = phis[t]
            if t + 1 == M:  # must stop at maturity
                A += ph.T @ ph
                b += ph.T @ (gamma * pays[M])
                continue
            ph1, g1 = phis[t + 1], pays[t + 1]
            stop = (g1 > 0) & (g1 >= ph1 @ w)
            A += ph.T @ (ph - gamma * (~stop)[:, None] * ph1)
            b += ph.T @ (gamma * np.where(stop, g1, 0.0))
        w_new = np.linalg.solve(A, b)
        change = float(np.linalg.norm(w_new - w) / max(np.linalg.norm(w_new), 1e-12))
        history.append(change)
        w = w_new
        if change < tol:
            converged = True
            break
    return _finish("LSPI", spec, paths, w, it, converged, history, M, degree, interactions, t0)


def fqi(spec: OptionSpec, paths: np.ndarray, degree: int = 3, interactions: bool = True,
        max_iter: int = 500, tol: float = 1e-5, ridge: float = 1e-8) -> RLResult:
    t0 = time.perf_counter()
    N = paths.shape[0]
    M, gamma, phis, pays = _setup(spec, paths, degree, interactions)
    k = phis[0].shape[1]
    A = ridge * N * np.eye(k) + sum(ph.T @ ph for ph in phis)
    w, history, converged = np.zeros(k), [], False
    for it in range(1, max_iter + 1):
        b = np.zeros(k)
        for t in range(M):
            target = pays[M] if t + 1 == M else np.maximum(pays[t + 1], phis[t + 1] @ w)
            b += phis[t].T @ (gamma * target)
        w_new = np.linalg.solve(A, b)
        change = float(np.linalg.norm(w_new - w) / max(np.linalg.norm(w_new), 1e-12))
        history.append(change)
        w = w_new
        if change < tol:
            converged = True
            break
    return _finish("FQI", spec, paths, w, it, converged, history, M, degree, interactions, t0)
