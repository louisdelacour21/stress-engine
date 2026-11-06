"""Cross-checks and grids: QuantLib bond reference, spot x vol grid of the put, asset-class cascade."""
import numpy as np
import pandas as pd

from .data import FACTORS
from .portfolio import IDX, NAV, Portfolio, bond_price


def quantlib_bond(y, coupon, years=10):
    """Price per 100 from QuantLib for an annual-coupon bond on a flat continuously compounded curve."""
    import QuantLib as ql
    t0 = ql.Date(15, 1, 2026)
    ql.Settings.instance().evaluationDate = t0
    sch = ql.Schedule(t0, t0 + ql.Period(years, ql.Years), ql.Period(ql.Annual), ql.NullCalendar(),
                      ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    bond = ql.FixedRateBond(0, 100.0, sch, [coupon], ql.Actual365Fixed())
    curve = ql.FlatForward(t0, y, ql.Actual365Fixed(), ql.Continuous, ql.Annual)
    return sum(c.amount() * curve.discount(c.date()) for c in bond.cashflows())


def bond_gap_pct_nominal(ys=(0.01, 0.025, 0.04, 0.06)):
    """Largest |own - QuantLib| in % of nominal; day-count differences (ACT/365 vs integer years) are the residual."""
    return max(abs(bond_price(y, y, 0.0, 0.03)[0] - quantlib_bond(y, 0.03)) for y in ys)


def spot_vol_grid(pf: Portfolio, spots=(-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20), vols=(-10, 0, 10, 20, 30, 50)):
    """Total portfolio P&L (% NAV) for S&P log-move x VIX points (other factors fixed at 0)."""
    rows = {}
    for dv in vols:
        col = []
        for ds in spots:
            x = np.zeros((1, len(FACTORS)))
            x[0, IDX["spx"]] = np.log1p(ds)
            x[0, IDX["vix"]] = np.log(max(pf.state["vix"] + dv, 1.0) / pf.state["vix"])
            col.append(pf.pnl(x).sum() / NAV * 100)
        rows[f"VIX {dv:+d}"] = col
    return pd.DataFrame(rows, index=[f"S&P {s:+.0%}" for s in spots]).T


CLASSES = {"rates": ["eur_2y", "eur_10y", "us_2y", "us_10y"], "credit": ["eur_sov_spread", "baa_spread"],
           "equity": ["sx5e", "spx"], "vol": ["vix"], "fx": ["eurusd"]}


def cascade(pf: Portfolio, x):
    """P&L by asset class applied alone, and the cross term = total - sum of the parts (% NAV)."""
    x = np.asarray(x, dtype=float).reshape(1, -1)
    out = {}
    for k, fs in CLASSES.items():
        m = np.zeros(len(FACTORS))
        m[[IDX[f] for f in fs]] = 1
        out[k] = pf.pnl(x * m).sum() / NAV * 100
    total = pf.pnl(x).sum() / NAV * 100
    out["cross_term"] = total - sum(out.values())
    out["total"] = total
    return out
