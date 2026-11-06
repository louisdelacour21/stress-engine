"""stress-engine run | scenario factor=value ... | reverse LOSS_PCT | bench"""
import argparse
import time

import numpy as np
import pandas as pd

from . import analysis, reverse
from . import covariance as cov
from . import portfolio as P
from .data import FACTORS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stress-engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="rebuild every output (make reproduce)")
    sc = sub.add_parser("scenario", help="custom partial shock, e.g. eur_10y=150 sx5e=-0.15 (bp for rates/spreads, log for others)")
    sc.add_argument("shocks", nargs="+")
    sc.add_argument("--regime", default="full", choices=list(cov.REGIMES))
    rv = sub.add_parser("reverse", help="most plausible move losing LOSS_PCT of NAV")
    rv.add_argument("loss_pct", type=float)
    sub.add_parser("bench", help="time the full revaluation on 100k random scenarios")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .run import main as run
        return run()
    pf = P.today()
    S = cov.estimate("full" if a.cmd != "scenario" else a.regime, 10)
    if a.cmd == "scenario":
        shock = {k: float(v) for k, v in (s.split("=") for s in a.shocks)}
        bad = set(shock) - set(FACTORS)
        if bad:
            raise SystemExit(f"unknown factors {sorted(bad)}; choose among {FACTORS}")
        x, _, d, cond = cov.propagate(S, shock)
        pnl = pf.pnl(x.to_numpy())[0] / P.NAV * 100
        print(pd.Series(pnl, [L["name"] for L in pf.lines]).round(3).to_string())
        print(f"total {pnl.sum():+.2f} % NAV | d = {d:.1f} | band {analysis.pnl_band(pf, S, shock)}"
              + (" | WARNING: ill-conditioned shock set" if cond > 1e6 else ""))
    elif a.cmd == "reverse":
        x, d = reverse.full(pf, S, a.loss_pct / 100)
        print(f"d* = {d:.1f}" if x is not None else "loss not reachable along this direction")
        if x is not None:
            print(pd.Series(x, FACTORS).round(3).to_string())
    else:
        x = np.random.default_rng(0).normal(size=(100_000, len(FACTORS))) * 0.01
        t = time.perf_counter()
        pf.pnl(x)
        print(f"100k full revaluations in {time.perf_counter() - t:.2f} s")


if __name__ == "__main__":
    main()
