"""Basis functions for continuation-value regression.

All take moneyness x = S / K. Column 0 is always the constant, so ridge regression can
leave the intercept unpenalised.

power      1, x, x^2, ..., x^d
laguerre   1, w L_0(x), ..., w L_{d-1}(x) with w = exp(-x/2) (Longstaff & Schwartz 2001)
hermite    probabilists' Hermite He_0..He_d of x standardised on the training sample
legendre   Legendre P_0..P_d of x mapped to [-1, 1] on the training sample
chebyshev  Chebyshev T_0..T_d of x mapped to [-1, 1] on the training sample

The scaling learned at fit time is reused when predicting on new paths.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev, hermite_e, laguerre, legendre

FAMILIES = ("power", "laguerre", "hermite", "legendre", "chebyshev")


def fit_scale(x: np.ndarray) -> dict:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    return {"mu": float(x.mean()), "sd": float(x.std()) or 1.0, "lo": lo, "hi": hi}


def design(x: np.ndarray, family: str, degree: int, scale: dict | None = None):
    """Return (design matrix with a constant first column, scale used)."""
    x = np.asarray(x, dtype=float)
    scale = scale or fit_scale(x)
    if family == "power":
        X = np.vander(x, degree + 1, increasing=True)
    elif family == "laguerre":
        w = np.exp(-x / 2)[:, None]
        X = np.hstack([np.ones((len(x), 1)), w * laguerre.lagvander(x, degree - 1)])
    elif family == "hermite":
        z = (x - scale["mu"]) / scale["sd"]
        X = hermite_e.hermevander(z, degree)
    elif family in ("legendre", "chebyshev"):
        z = 2 * (x - scale["lo"]) / (scale["hi"] - scale["lo"]) - 1
        X = (legendre.legvander if family == "legendre" else chebyshev.chebvander)(z, degree)
    else:
        raise ValueError(f"unknown basis family {family!r}; choose from {FAMILIES}")
    return X, scale
