"""Covariance of the ten factors (1-day changes), regimes, and Gaussian conditional propagation of a partial shock."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from .data import changes

REGIMES = {  # inclusive date ranges; stressed = union
    "full": [("2005-01-01", "2026-12-31")],
    "stressed": [("2008-09-01", "2009-06-30"), ("2011-07-01", "2012-01-31"), ("2020-02-15", "2020-06-30")],
    "post2021": [("2021-01-01", "2026-12-31")],
}


def regime(dx: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.concat([dx.loc[a:b] for a, b in REGIMES[name]])


def ewma(dx: pd.DataFrame, lam=0.97):
    """EWMA mean-zero covariance, then Ledoit-Wolf shrinkage towards a scaled identity on the standardised data."""
    X = dx.to_numpy()
    w = lam ** np.arange(len(X))[::-1]
    S = (X * w[:, None]).T @ X / w.sum()
    sd = np.sqrt(np.diag(S))
    lw = LedoitWolf().fit(X / sd)  # shrink the correlation, keep the EWMA variances
    return pd.DataFrame(np.outer(sd, sd) * lw.covariance_, dx.columns, dx.columns)


def horizon(S1: pd.DataFrame, h=10) -> pd.DataFrame:
    """Square-root-of-time scaling: ignores autocorrelation and vol clustering inside the window (limit)."""
    return S1 * h


def propagate(S: pd.DataFrame, shock: dict, mu=None):
    """Conditional mean and covariance of the unshocked factors given the shocked ones, plus Mahalanobis d.

    Returns (full vector with shocks imposed, conditional covariance, d, cond number of S_ss)."""
    s = list(shock)
    u = [c for c in S.columns if c not in shock]
    mu = pd.Series(0.0, S.columns) if mu is None else mu
    Sss, Sus = S.loc[s, s].to_numpy(), S.loc[u, s].to_numpy()
    cond = np.linalg.cond(Sss)
    v = np.array([shock[c] for c in s]) - mu[s].to_numpy()
    sol = np.linalg.solve(np.linalg.cholesky(Sss).T, np.linalg.solve(np.linalg.cholesky(Sss), v))
    x = mu.copy()
    x[s] = [shock[c] for c in s]
    x[u] = mu[u].to_numpy() + Sus @ sol
    C = pd.DataFrame(S.loc[u, u].to_numpy() - Sus @ np.linalg.solve(Sss, Sus.T), u, u)
    d = float(np.sqrt(v @ sol))
    return x, C, d, cond


def p_directional(d: float) -> float:
    """P(score along the shock direction >= d) under N(0,1): Phi(-d), not chi-square."""
    return float(norm.sf(d))


def estimate(regime_name="full", h=10):
    dx = changes()
    return horizon(ewma(regime(dx, regime_name)), h)
