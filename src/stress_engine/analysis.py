"""Regime comparison, P&L bands from the conditional covariance, factor and LQD diagnostics."""
import numpy as np
import pandas as pd

from . import covariance as cov
from .data import FACTORS, changes, lqd, panel
from .portfolio import NAV, Portfolio


def regimes_table(pf: Portfolio, hypo: dict) -> pd.DataFrame:
    """P&L (% NAV) of each hypothetical scenario with the unshocked factors propagated under each regime's covariance."""
    dx, rows = changes(), []
    for r in cov.REGIMES:
        S = cov.horizon(cov.ewma(cov.regime(dx, r)), 10)
        for k, sh in hypo.items():
            x, _, d, cond = cov.propagate(S, sh)
            rows.append({"regime": r, "scenario": k, "pnl_pct_nav": pf.pnl(x.to_numpy())[0].sum() / NAV * 100,
                         "d": d, "cond_Sss": cond})
    return pd.DataFrame(rows)


def pnl_band(pf: Portfolio, S, shock: dict, n=5000, seed=3):
    """5 / 50 / 95 % quantiles of the full-revaluation P&L (% NAV) when the unshocked factors are drawn Gaussian."""
    x, C, _, _ = cov.propagate(S, shock)
    u = list(C.index)
    draws = np.tile(x.to_numpy(), (n, 1))
    idx = [FACTORS.index(c) for c in u]
    draws[:, idx] = np.random.default_rng(seed).multivariate_normal(x[u].to_numpy(), C.to_numpy(), n, check_valid="ignore")
    p = pf.pnl(draws).sum(1) / NAV * 100
    return {k: round(float(v), 3) for k, v in zip(("p05", "p50", "p95"), np.percentile(p, [5, 50, 95]))}


def lqd_tracking(h=10) -> dict:
    """How well the Baa spread (BAA10Y) explains LQD: regression of 10d LQD log-returns on -duration * spread changes."""
    p = panel()
    q = lqd().reindex(p.index).dropna()
    dl = np.log(q).diff(h).dropna()
    ds = (p["baa_spread"].reindex(q.index).diff(h) / 100).reindex(dl.index)  # percent -> decimal
    ok = ds.notna()
    X = np.column_stack([np.ones(ok.sum()), -ds[ok]])
    b = np.linalg.lstsq(X, dl[ok], rcond=None)[0]
    res = dl[ok] - X @ b
    return {"implied_duration": float(b[1]), "r2": float(1 - res.var() / dl[ok].var()),
            "tracking_error_10d_pct": float(res.std() * 100)}


def factor_stats() -> pd.DataFrame:
    """Daily distribution of each factor: shows the fat tails that the Gaussian propagation ignores."""
    dx = changes()
    return pd.DataFrame({"sd_1d": dx.std(), "skew": dx.skew(), "excess_kurtosis": dx.kurt(),
                         "worst_1d": dx.min(), "best_1d": dx.max()}).round(4)
