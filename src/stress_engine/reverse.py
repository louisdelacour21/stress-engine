"""Reverse stress: most plausible (smallest Mahalanobis distance) factor move that loses L* of NAV."""
import numpy as np

from .portfolio import NAV, Portfolio
from .sensitivities import greeks


def linear(pf: Portfolio, S, loss_pct, mu=None):
    """Closed form on the delta (gradient) approximation: x* = -L Sa / (a'Sa) (mu = 0), d* = L / sqrt(a'Sa)."""
    a = greeks(pf)[0].sum(0)
    S = np.asarray(S)
    L = loss_pct * NAV
    sa = S @ a
    return -L * sa / (a @ sa), L / np.sqrt(a @ sa)


def full(pf: Portfolio, S, loss_pct, tol=1e-6):
    """Same direction, scale found by bisection so that the FULL revaluation loses exactly L*."""
    x0, _ = linear(pf, S, loss_pct)
    L = loss_pct * NAV
    f = lambda k: -pf.pnl(k * x0).sum() - L
    lo, hi = 0.0, 1.0
    while f(hi) < 0 and hi < 64:
        hi *= 2
    if f(hi) < 0:
        return None, np.inf  # not reachable along this direction
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    x = hi * x0
    return x, float(np.sqrt(x @ np.linalg.solve(np.asarray(S), x)))


def curve(pf, S, levels=range(2, 21, 2)):
    return [(L, linear(pf, S, L / 100)[1], full(pf, S, L / 100)[1]) for L in levels]
