"""Rolling out-of-sample test of the conditional propagation on non-overlapping h-day periods.

For each period t: Sigma estimated on data strictly before t, condition on the realised change of the pilot factor,
predict the others; PIT_k = Phi((x_k - m_k)/s_k). Under a correct model PIT ~ U(0,1)."""
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

from .covariance import ewma, propagate
from .data import changes


def run(pilot="sx5e", h=10, start="2007-12-31", lam=0.97, df=None):
    dx1 = changes(df, 1)
    dxh = changes(df, h)
    idx = dxh.index[::h]  # non-overlapping windows
    idx = idx[idx >= start]
    rows = []
    for t in idx:
        past = dx1.loc[: t - pd.Timedelta(days=h * 2)]  # strictly before the window opens
        if len(past) < 500:
            continue
        S = ewma(past, lam) * h
        x = dxh.loc[t]
        m, C, d, _ = propagate(S, {pilot: x[pilot]})
        sd = np.sqrt(np.diag(C))
        u = [c for c in S.columns if c != pilot]
        z = (x[u].to_numpy() - m[u].to_numpy()) / sd
        rows.append(pd.Series(norm.cdf(z), u, name=t))
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - w, c + w


def berkowitz(pit):
    """LR test that z = Phi^-1(PIT) is iid N(0,1) against AR(1) with free mean and variance."""
    z = norm.ppf(np.clip(pit, 1e-6, 1 - 1e-6))
    y, x = z[1:], z[:-1]
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = np.mean((y - X @ b) ** 2)
    ll1 = -0.5 * len(y) * (np.log(2 * np.pi * s2) + 1)
    ll0 = -0.5 * np.sum(y**2) - 0.5 * len(y) * np.log(2 * np.pi)
    lr = 2 * (ll1 - ll0)
    from scipy.stats import chi2
    return lr, float(chi2.sf(lr, 3))


def summary(pit: pd.DataFrame):
    out = {}
    for c in pit:
        p = pit[c].dropna().to_numpy()
        k = int(((p > 0.05) & (p < 0.95)).sum())  # 90 % band hits
        lo, hi = wilson(k, len(p))
        out[c] = dict(n=len(p), ks_p=float(kstest(p, "uniform").pvalue), berkowitz_p=berkowitz(p)[1],
                      cover90=k / len(p), cover90_lo=lo, cover90_hi=hi)
    return pd.DataFrame(out).T
