"""Stress lab: combined shocks in trader units, full revaluation, driver attribution, and VaR/ES along the shock path.

A spec maps drivers to sizes (equity in %, VIX in points, rates and spreads in bp, EURUSD in %). Drivers set to 0 are not
imposed; the other factors are completed by conditional expectation (covariance S) or left at 0. Risk is measured from
the shocked point: P&L(x0 + dx) - P&L(x0), dx ~ N(0, S) or Student-t (unit variance), S a 10-day covariance."""
import numpy as np
import pandas as pd

from . import covariance as cov
from .data import FACTORS

LABEL = {"equity_pct": "Equity (SX5E and SPX), %", "vix_pts": "VIX, points", "eur_rates_bp": "EUR rates 2y+10y, bp",
         "us_rates_bp": "US rates 2y+10y, bp", "eur_sov_spread_bp": "EUR sovereign spread, bp",
         "baa_spread_bp": "US Baa credit spread, bp", "eurusd_pct": "EURUSD (USD per EUR), %"}
# illustrative hypotheticals, not forecasts
PRESETS = {
    "Custom": {},
    "Equity crash + vol spike": {"equity_pct": -25, "vix_pts": 25, "baa_spread_bp": 100},
    "Rates shock (2022 style)": {"eur_rates_bp": 200, "us_rates_bp": 250, "equity_pct": -15, "vix_pts": 10},
    "Stagflation": {"eur_rates_bp": 150, "us_rates_bp": 150, "equity_pct": -15, "vix_pts": 15, "baa_spread_bp": 80},
    "Euro fragmentation": {"eur_sov_spread_bp": 150, "equity_pct": -10, "vix_pts": 10},
    "Strong dollar": {"eurusd_pct": -15, "us_rates_bp": 100},
}


def factor_shock(state: dict, spec: dict) -> dict:
    """Spec -> shocks on the model factors (bp, or log-change), only for non-zero drivers."""
    g = spec.get
    out = {}
    if g("equity_pct"):
        out["sx5e"] = out["spx"] = float(np.log1p(g("equity_pct") / 100))
    if g("vix_pts"):
        out["vix"] = float(np.log(max(state["vix"] + g("vix_pts"), 1.0) / state["vix"]))
    if g("eur_rates_bp"):
        out["eur_2y"] = out["eur_10y"] = float(g("eur_rates_bp"))
    if g("us_rates_bp"):
        out["us_2y"] = out["us_10y"] = float(g("us_rates_bp"))
    if g("eur_sov_spread_bp"):
        out["eur_sov_spread"] = float(g("eur_sov_spread_bp"))
    if g("baa_spread_bp"):
        out["baa_spread"] = float(g("baa_spread_bp"))
    if g("eurusd_pct"):
        out["eurusd"] = float(np.log1p(g("eurusd_pct") / 100))
    return out


def shock_vector(state, spec, S, complete=True):
    """Full 10-factor vector, Mahalanobis d of the imposed block, condition number of its covariance."""
    f = factor_shock(state, spec)
    if not f:
        return np.zeros(len(FACTORS)), 0.0, 1.0
    x, _, d, cond = cov.propagate(S, f)
    if not complete:
        x = pd.Series(f).reindex(S.columns).fillna(0.0)
    return x.reindex(FACTORS).to_numpy(float), d, cond


def draws(S, n=20000, nu=None, seed=0):
    """Factor moves, common random numbers across calls; nu=None Gaussian, else Student-t with unit variance."""
    z = np.random.default_rng(seed).standard_normal((n, len(S)))
    if nu:
        z = z * np.sqrt((nu - 2) / np.random.default_rng(seed + 1).chisquare(nu, n))[:, None]
    return z @ np.linalg.cholesky(S.to_numpy()).T


def risk(pf, x0, dx, alpha=0.99):
    """VaR and ES (positive losses, EUR) of the horizon P&L started from the shocked point x0, ES split by line."""
    x0 = np.asarray(x0, float)[None, :]
    lines = pf.values(x0 + dx) - pf.values(x0)
    tot = lines.sum(1)
    q = np.quantile(tot, 1 - alpha)
    tail = tot <= q
    return {"var": float(-q), "es": float(-tot[tail].mean()), "es_by_line": -lines[tail].mean(0)}


def attribution(pf, spec, S, complete=True):
    """P&L (EUR) of each driver alone, then what the combination adds (cross terms, plus completion if on)."""
    total = pf.pnl(shock_vector(pf.state, spec, S, complete)[0])[0].sum()
    alone = {k: pf.pnl(shock_vector(pf.state, {k: v}, S, False)[0])[0].sum() for k, v in spec.items() if v}
    return pd.Series({**alone, "cross terms + completion": total - sum(alone.values()), "TOTAL": total})


def ramp(pf, spec, S, steps=11, n=10000, nu=None, complete=True):
    """Shock scaled from 0 to 100 %: P&L by line, VaR and ES (from each shocked point) at every step."""
    dx, rows = draws(S, n, nu), []
    for k in np.linspace(0, 1, steps):
        x = shock_vector(pf.state, {a: v * k for a, v in spec.items()}, S, complete)[0]
        r = risk(pf, x, dx)
        rows.append({"severity_pct": round(k * 100), **dict(zip((L["name"] for L in pf.lines), pf.pnl(x)[0])),
                     "VaR": r["var"], "ES": r["es"]})
    return pd.DataFrame(rows).set_index("severity_pct")


def grid(pf, spec, k1, v1, k2, v2, S, complete=True):
    """Total P&L (EUR) over two drivers; the other drivers stay at their spec values."""
    return pd.DataFrame([[pf.pnl(shock_vector(pf.state, {**spec, k1: a, k2: b}, S, complete)[0])[0].sum() for b in v2] for a in v1],
                        index=v1, columns=v2)
