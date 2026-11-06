"""Taylor approximations of the full revaluation, by central finite differences on `Portfolio.values`.

order 1: delta/DV01/CS01/vega only; order 2: adds the diagonal gamma/convexity and the cross terms (Hessian).
`only` restricts the factors that enter the expansion (e.g. a Δ-Γ approximation without vega drops the VIX)."""
import numpy as np

from .data import FACTORS
from .portfolio import IDX, NAV, Portfolio

STEP = {True: 1.0, False: 1e-3}  # bump: 1 bp for rates/spreads, 0.1 % for log-changes


def _step(f):
    return 1.0 if f.endswith(("2y", "10y")) or "spread" in f else 1e-3


def greeks(pf: Portfolio):
    """Gradient (lines, 10) and Hessian (lines, 10, 10) of the line values at zero shock."""
    n, h = len(FACTORS), np.array([_step(f) for f in FACTORS])
    z = np.zeros(n)

    def v(*pairs):
        x = z.copy()
        for i, s in pairs:
            x[i] += s
        return pf.values(x)[0]

    v0 = v()
    g = np.zeros((len(v0), n))
    H = np.zeros((len(v0), n, n))
    for i in range(n):
        up, dn = v((i, h[i])), v((i, -h[i]))
        g[:, i] = (up - dn) / (2 * h[i])
        H[:, i, i] = (up - 2 * v0 + dn) / h[i] ** 2
        for j in range(i):
            H[:, i, j] = H[:, j, i] = (v((i, h[i]), (j, h[j])) - v((i, h[i]), (j, -h[j]))
                                       - v((i, -h[i]), (j, h[j])) + v((i, -h[i]), (j, -h[j]))) / (4 * h[i] * h[j])
    return g, H


def approx(pf: Portfolio, x, order=2, only=None):
    """Approximate P&L per line in EUR, shape (n, lines)."""
    g, H = greeks(pf)
    x = np.atleast_2d(np.asarray(x, dtype=float)).copy()
    if only is not None:
        keep = np.zeros(len(FACTORS), bool)
        keep[[IDX[f] for f in only]] = True
        x[:, ~keep] = 0.0
    p = x @ g.T
    if order >= 2:
        p = p + 0.5 * np.einsum("ni,lij,nj->nl", x, H, x)
    return p


def error_alert(pf: Portfolio, x, order=2):
    """Approximation error e = approx - full (total, EUR) and the two alert tests of the fiche."""
    full = pf.pnl(x).sum(1)
    e = approx(pf, x, order).sum(1) - full
    return e, (np.abs(e) > 0.10 * np.abs(full)) | (np.abs(e) > 0.0025 * NAV)
