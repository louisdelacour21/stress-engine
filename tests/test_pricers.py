import numpy as np
import pytest

from stress_engine import portfolio as P
from stress_engine.portfolio import bond_price, bs
from stress_engine.sensitivities import approx, error_alert, greeks


def b(y):  # fiche bond: 10y, coupon 2.5 %, flat curve
    return bond_price(y, y, 0.0, 0.025)[0]


def test_bond_control_case():
    assert b(0.025) == pytest.approx(99.72, abs=0.01)
    assert b(0.035) - b(0.025) == pytest.approx(-8.53, abs=0.01)
    assert b(0.050) - b(0.025) == pytest.approx(-19.89, abs=0.01)
    d, c = 8.97, 86.0  # fiche duration and convexity
    p0 = b(0.025)
    assert p0 * (-d * 0.01 + c * 0.0001 / 2) == pytest.approx(-8.52, abs=0.02)
    assert p0 * (-d * 0.025 + c * 0.025**2 / 2) == pytest.approx(-19.68, abs=0.05)


def test_put_control_case():
    put = lambda S, s: bs(S, 95.0, 0.25, 0.0, s, "put")
    assert put(100.0, 0.20) == pytest.approx(1.89, abs=0.01)
    assert put(85.0, 0.32) - put(100.0, 0.20) == pytest.approx(10.18, abs=0.02)
    assert put(70.0, 0.45) - put(100.0, 0.20) == pytest.approx(23.85, abs=0.02)


def test_put_call_parity():
    S, K, T, r, s = 100.0, 105.0, 0.25, 0.03, 0.22
    assert bs(S, K, T, r, s, "call") - bs(S, K, T, r, s, "put") == pytest.approx(S - K * np.exp(-r * T))


def test_portfolio_starts_flat_and_taylor_is_third_order():
    pf = P.today()
    assert np.allclose(pf.pnl(np.zeros(10)), 0)
    g, H = greeks(pf)
    x = np.zeros(10)
    x[P.IDX["us_10y"]], x[P.IDX["us_2y"]] = 1.0, 1.0  # 1 bp parallel
    e = []
    for k in (50, 100):
        s = x * k
        e.append(abs(approx(pf, s).sum() - pf.pnl(s).sum()))
    # error of a 2nd-order expansion scales like k^3 (slope 3 in log-log)
    assert np.log2(e[1] / e[0]) == pytest.approx(3, abs=0.4)


def test_error_alert_flags_a_large_equity_crash():
    pf = P.today()
    x = np.zeros((1, 10))
    x[0, P.IDX["spx"]], x[0, P.IDX["vix"]] = -0.4, 1.0
    assert error_alert(pf, x, order=1)[1][0]
