"""Nine synthetic lines and their full revaluation, vectorised over shocks.

A shock is an (n, 10) array in factor units (bp for rates and spreads, log-change otherwise), columns in data.FACTORS
order. `values(x)` returns the EUR value of each line, shape (n, lines). Bonds are priced on a zero curve with two
pillars (2y, 10y), linear in maturity and flat outside, continuous compounding, annual coupons set at par today."""
from dataclasses import dataclass

import numpy as np
import yaml
from scipy.stats import norm

from .data import FACTORS, ROOT, panel

IDX = {n: i for i, n in enumerate(FACTORS)}
CFG = yaml.safe_load((ROOT / "config" / "portfolio.yaml").read_text(encoding="utf-8"))
NAV = CFG["nav_eur"]


def bond_price(y2, y10, spread, coupon, years=10):
    """Price per 100 of an annual-coupon bond; y2, y10, spread in decimals (arrays broadcast)."""
    t = np.arange(1, years + 1)
    y = np.interp(t, [2, 10], [0, 1])[:, None] * (np.asarray(y10) - np.asarray(y2)).reshape(1, -1) + np.asarray(y2).reshape(1, -1)
    y = y + np.asarray(spread).reshape(1, -1)
    cf = np.full(years, coupon * 100.0)
    cf[-1] += 100.0
    return (cf[:, None] * np.exp(-y * t[:, None])).sum(0)


def bs(S, K, T, r, sig, right):
    d1 = (np.log(S / K) + (r + sig * sig / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if right == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


@dataclass
class Portfolio:
    state: dict  # factor levels today (percent, index, ...)
    lines: list | None = None  # default: config/portfolio.yaml

    def __post_init__(self):
        s = self.state
        self.lines = self.lines or CFG["lines"]
        self.coupon = {}
        for L in self.lines:
            if L["kind"] == "bond":
                y2, y10 = s[f"{L['curve']}_2y"] / 100, s[f"{L['curve']}_10y"] / 100
                sp = s[L["spread"]] / 100 if L["spread"] else 0.0
                # par coupon on the initial curve: price 100 at inception
                a = bond_price(y2, y10, sp, 0.0)[0]
                ann = bond_price(y2, y10, sp, 1.0)[0] - a
                self.coupon[L["name"]] = (100 - a) / ann

    def _fx(self, x):  # EUR per USD
        return 1 / (self.state["eurusd"] * np.exp(x[:, IDX["eurusd"]]))

    def values(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        s, fx0, fx = self.state, 1 / self.state["eurusd"], self._fx(x)
        cols = []
        for L in self.lines:
            n, k = L["nominal"], L["kind"]
            if k == "bond":
                c = L["curve"]
                y2 = s[f"{c}_2y"] / 100 + x[:, IDX[f"{c}_2y"]] / 1e4
                y10 = s[f"{c}_10y"] / 100 + x[:, IDX[f"{c}_10y"]] / 1e4
                sp = s[L["spread"]] / 100 + x[:, IDX[L["spread"]]] / 1e4 if L["spread"] else 0.0
                v = n * bond_price(y2, y10, sp, self.coupon[L["name"]]) / 100
            elif k == "future":
                v = n * np.exp(L.get("beta", 1.0) * x[:, IDX[L["index"]]])  # beta: single stock proxied by its index
            elif k == "fxfwd":
                T = L["months"] / 12
                rE = s["eur_2y"] / 100 + x[:, IDX["eur_2y"]] / 1e4
                rU = s["us_2y"] / 100 + x[:, IDX["us_2y"]] / 1e4
                spot_usd = fx  # EUR per USD
                strike = fx0 * np.exp((s["eur_2y"] - s["us_2y"]) / 100 * T)
                v = n * (strike - spot_usd * np.exp((rE - rU) * T)) * np.exp(-rE * T)
            else:  # option on the S&P 500, vol = VIX
                T = L["months"] / 12
                S0 = s["spx"]
                r = s["us_2y"] / 100 + x[:, IDX["us_2y"]] / 1e4
                sig = s["vix"] / 100 * np.exp(x[:, IDX["vix"]])
                S = S0 * np.exp(x[:, IDX["spx"]])
                units = n / S0
                v = units * bs(S, L["moneyness"] * S0, T, r, sig, L["right"])
            if L["ccy"] == "USD" and k not in ("fxfwd",):
                v = v * fx
            cols.append(v)
        return np.column_stack(cols)

    def pnl(self, x) -> np.ndarray:
        """Full-revaluation P&L per line in EUR (value after shock minus value today)."""
        x = np.atleast_2d(x)
        return self.values(x) - self.values(np.zeros((1, len(FACTORS))))


def today() -> Portfolio:
    return Portfolio(panel().iloc[-1].to_dict())
