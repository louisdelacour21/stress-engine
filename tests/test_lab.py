import json

import numpy as np
import pandas as pd
import pytest

from stress_engine import lab
from stress_engine import portfolio as P
from stress_engine.data import ROOT

SNAP = json.loads((ROOT / "outputs" / "snapshot.json").read_text())
S = pd.DataFrame(SNAP["cov"]).reindex(index=SNAP["factors"], columns=SNAP["factors"])
SS = pd.DataFrame(SNAP["cov_stressed"]).reindex(index=SNAP["factors"], columns=SNAP["factors"])
PF = P.Portfolio(SNAP["state"])
SPEC = {"equity_pct": -20, "vix_pts": 20, "us_rates_bp": 100}


def test_shock_units_and_zero_spec():
    f = lab.factor_shock(PF.state, SPEC)
    assert f["spx"] == pytest.approx(np.log(0.8)) and f["us_2y"] == f["us_10y"] == 100
    assert f["vix"] == pytest.approx(np.log((PF.state["vix"] + 20) / PF.state["vix"]))
    assert not lab.shock_vector(PF.state, {}, S)[0].any()


def test_attribution_adds_up_and_equals_revaluation():
    a = lab.attribution(PF, SPEC, S)
    assert a.drop("TOTAL").sum() == pytest.approx(a["TOTAL"])
    assert a["TOTAL"] == pytest.approx(PF.pnl(lab.shock_vector(PF.state, SPEC, S)[0])[0].sum())


def test_risk_ordering_and_ramp_endpoints():
    r = lab.risk(PF, np.zeros(10), lab.draws(S, 20000))
    assert r["es"] >= r["var"] > 0 and r["es_by_line"].sum() == pytest.approx(r["es"])
    assert lab.risk(PF, np.zeros(10), lab.draws(SS, 20000))["var"] > r["var"]  # stressed regime is riskier
    assert lab.risk(PF, np.zeros(10), lab.draws(S, 20000, nu=5))["es"] > r["es"] * 0.9  # fat tails do not shrink the tail
    rp = lab.ramp(PF, SPEC, S, steps=3, n=4000)
    assert rp.iloc[0, :9].abs().max() < 1e-6
    x = lab.shock_vector(PF.state, SPEC, S)[0]
    assert rp.iloc[-1, :9].to_numpy() == pytest.approx(PF.pnl(x)[0])
