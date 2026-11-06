import numpy as np
import pandas as pd
import pytest

from stress_engine import covariance as C
from stress_engine import portfolio as P
from stress_engine import reverse as R
from stress_engine.backtest import berkowitz, wilson


def test_conditional_matches_bivariate_formula():
    S = pd.DataFrame([[4.0, 1.2], [1.2, 1.0]], ["a", "b"], ["a", "b"])
    x, cv, d, _ = C.propagate(S, {"a": 2.0})
    assert x["b"] == pytest.approx(0.6) and cv.loc["b", "b"] == pytest.approx(1 - 1.2**2 / 4)
    assert d == pytest.approx(1.0) and C.p_directional(1.0) == pytest.approx(0.1587, abs=1e-3)


def test_reverse_linear_is_on_the_loss_level_and_full_is_farther():
    pf, S = P.today(), C.estimate("full", 10)
    x, d = R.linear(pf, S, 0.05)
    assert d == pytest.approx(np.sqrt(x @ np.linalg.solve(S.to_numpy(), x)))
    assert R.full(pf, S, 0.05)[1] > d * 0.9


def test_wilson_and_berkowitz_on_iid_uniform():
    lo, hi = wilson(90, 100)
    assert lo < 0.9 < hi
    assert berkowitz(np.random.default_rng(1).uniform(size=800))[1] > 0.01


def test_bond_agrees_with_quantlib_and_cascade_adds_up():
    pytest.importorskip("QuantLib")
    from stress_engine import checks
    assert checks.bond_gap_pct_nominal() < 0.02  # % of nominal; ACT/365 vs whole-year discounting
    pf = P.today()
    c = checks.cascade(pf, np.r_[50, 50, 10, 50, 50, 5, 0.2, -0.05, -0.2, -0.2])
    assert sum(v for k, v in c.items() if k != "total") == pytest.approx(c["total"])


def test_reproduce_writes_outputs():
    from stress_engine import run
    run.main()
    for f in ("scenarios.csv", "reverse.csv", "backtest.csv", "spot_vol_grid.csv", "cascade.csv", "summary.html", "meta.json"):
        assert (run.OUT / f).stat().st_size > 0
