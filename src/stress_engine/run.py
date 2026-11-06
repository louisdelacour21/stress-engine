"""`make reproduce`: derived outputs only (raw series are not redistributed) + a one-page HTML summary."""
import json
import time

import numpy as np
import pandas as pd

from . import backtest, checks, data, reverse, scenarios
from . import covariance as cov
from . import portfolio as P
from .sensitivities import error_alert

OUT = data.ROOT / "outputs"


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    pf, S = P.today(), cov.estimate("full", 10)
    names = [L["name"] for L in pf.lines]
    rows = []
    for k, sh in {**{f"hist:{k}": v.to_dict() for k, v in scenarios.historical().iterrows()},
                  **{f"hypo:{k}": v for k, v in scenarios.hypothetical().items()}}.items():
        partial = {f: v for f, v in sh.items() if f in data.FACTORS and not np.isnan(v)}
        full_vec = pd.Series(partial).reindex(data.FACTORS)
        x_hist = full_vec.fillna(0.0).to_numpy()
        if k.startswith("hist:"):
            x, d = x_hist, np.nan  # historical: all ten factors observed, no propagation
        else:
            m, _, d, cond = cov.propagate(S, partial)
            x = m.to_numpy()
        pnl = pf.pnl(x)[0]
        e, alert = error_alert(pf, x, 2)
        rows.append({"scenario": k, "pnl_pct_nav": pnl.sum() / P.NAV * 100, "mahalanobis_d": d,
                     "taylor2_error_pct_nav": float(e[0]) / P.NAV * 100, "alert": bool(alert[0]),
                     **{n: v / 1e6 for n, v in zip(names, pnl)}})
    pd.DataFrame(rows).to_csv(OUT / "scenarios.csv", index=False)
    pd.DataFrame(reverse.curve(pf, S), columns=["loss_pct_nav", "d_linear", "d_full"]).to_csv(OUT / "reverse.csv", index=False)
    checks.spot_vol_grid(pf).round(3).to_csv(OUT / "spot_vol_grid.csv")
    hist = scenarios.historical()
    pd.DataFrame({k: checks.cascade(pf, hist.loc[k].fillna(0).to_numpy()) for k in hist.index}).T.round(3).to_csv(OUT / "cascade.csv")
    bt = pd.concat({p: backtest.summary(backtest.run(p)) for p in ("sx5e", "us_10y")})
    bt.to_csv(OUT / "backtest.csv")
    meta = {"bond_gap_pct_nominal_vs_quantlib": round(checks.bond_gap_pct_nominal(), 5), "panel_fingerprint": data.fingerprint(data.panel()), "seconds": round(time.time() - t0, 1)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    sc_df = pd.read_csv(OUT / "scenarios.csv")
    html = ("<!doctype html><meta charset=utf-8><title>Stress engine summary</title><h1>Stress engine: morning summary</h1>"
            "<p>Self-review, synthetic 100 MEUR portfolio. Instantaneous shocks (no theta). Gaussian propagation: see backtest.</p>"
            + sc_df[["scenario", "pnl_pct_nav", "mahalanobis_d", "taylor2_error_pct_nav", "alert"]].round(2).to_html(index=False)
            + "<h2>Spot x vol grid (% NAV)</h2>" + pd.read_csv(OUT / "spot_vol_grid.csv", index_col=0).to_html()
            + "<h2>Asset-class cascade (% NAV)</h2>" + pd.read_csv(OUT / "cascade.csv", index_col=0).to_html()
            + "<h2>Reverse stress</h2>" + pd.read_csv(OUT / "reverse.csv").round(2).to_html(index=False)
            + "<h2>Propagation backtest</h2>" + bt.round(3).to_html())
    (OUT / "summary.html").write_text(html, encoding="utf-8")
    print(f"done in {meta['seconds']} s")


if __name__ == "__main__":
    main()
