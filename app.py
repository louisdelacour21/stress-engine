"""Streamlit demo: live revaluation of an editable portfolio (tab 1) + precomputed outputs/ (other tabs). No raw market data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from stress_engine import lab  # noqa: E402
from stress_engine import portfolio as P  # noqa: E402

OUTD = Path(__file__).parent / "outputs"
st.set_page_config(page_title="Stress engine", layout="wide")
st.title("Stress engine")
st.caption("Self-review. Instantaneous shocks, no raw market data redistributed. Stress lab and Crisis scenarios revalue YOUR positions live (default: a synthetic 100 MEUR book); the Ref tabs are precomputed.")
SNAP = json.loads((OUTD / "snapshot.json").read_text())
FACT = SNAP["factors"]
sc = pd.read_csv(OUTD / "scenarios.csv")
COLS = ["name", "kind", "ccy", "nominal", "curve", "spread", "index", "beta", "right", "months", "moneyness"]
EX = {"name": "My line", "kind": "future", "ccy": "EUR", "nominal": 5e6, "curve": None, "spread": None, "index": "sx5e",
      "beta": 1.0, "right": None, "months": None, "moneyness": None}


def build(df):
    """Rows of the editor -> portfolio lines (validated); raises ValueError with a readable message."""
    lines = []
    for i, r in df.iterrows():
        L = {"name": f"{i + 1}. {r['name']}", "kind": r["kind"], "ccy": r["ccy"], "nominal": float(r["nominal"])}
        if r["kind"] == "bond":
            L.update(curve=r["curve"], spread=None if r["spread"] in (None, "none") or pd.isna(r["spread"]) else r["spread"])
            if L["curve"] not in ("eur", "us"):
                raise ValueError(f"row {i + 1}: a bond needs curve eur or us")
        elif r["kind"] == "future":
            L.update(index=r["index"], beta=1.0 if pd.isna(r["beta"]) else float(r["beta"]))
            if L["index"] not in ("sx5e", "spx"):
                raise ValueError(f"row {i + 1}: an equity line needs index sx5e or spx")
        elif r["kind"] == "fxfwd":
            L["months"] = float(r["months"]) if not pd.isna(r["months"]) else 3.0
        elif r["kind"] == "option":
            L.update(right=r["right"], months=float(r["months"]), moneyness=float(r["moneyness"]))
            if L["right"] not in ("put", "call") or pd.isna(r["months"]) or pd.isna(r["moneyness"]):
                raise ValueError(f"row {i + 1}: an S&P option needs right, months and moneyness")
        else:
            raise ValueError(f"row {i + 1}: unknown kind")
        lines.append(L)
    return lines



def lab_call(lines, spec, regime, nu, complete, steps=11, n=10000):
    """Everything the lab tab shows, cached on its inputs."""
    spec, pf = dict(spec), P.Portfolio(SNAP["state"], lines)
    S = pd.DataFrame(SNAP["cov" if regime == "Normal" else "cov_stressed"]).reindex(index=FACT, columns=FACT)
    x, d, cond = lab.shock_vector(pf.state, spec, S, complete)
    dx = lab.draws(S, 2 * n, nu)
    return {"x": x, "d": d, "cond": cond, "pnl": pf.pnl(x)[0], "names": [L["name"] for L in lines],
            "before": lab.risk(pf, np.zeros(len(FACT)), dx), "after": lab.risk(pf, x, dx),
            "attr": lab.attribution(pf, spec, S, complete), "ramp": lab.ramp(pf, spec, S, steps, n, nu, complete)}


lab_cached = st.cache_data(lab_call, show_spinner="Revaluing...")
SLIDERS = {"equity_pct": (-50, 20, 1), "vix_pts": (-10, 60, 1), "eur_rates_bp": (-200, 300, 10), "us_rates_bp": (-200, 300, 10),
           "eur_sov_spread_bp": (-50, 300, 10), "baa_spread_bp": (-50, 300, 10), "eurusd_pct": (-25, 25, 1)}
GRIDV = {"equity_pct": [-40, -30, -20, -10, 0, 10], "vix_pts": [0, 10, 20, 30, 50], "eur_rates_bp": [-100, 0, 100, 200, 300],
         "us_rates_bp": [-100, 0, 100, 200, 300], "eur_sov_spread_bp": [0, 50, 100, 200, 300], "baa_spread_bp": [0, 50, 100, 200, 300],
         "eurusd_pct": [-20, -10, 0, 10, 20]}
for k in SLIDERS:
    st.session_state.setdefault(f"s_{k}", 0)


def apply_preset():
    for k in SLIDERS:
        st.session_state[f"s_{k}"] = lab.PRESETS[st.session_state["preset"]].get(k, 0)


with st.expander("1. Your portfolio (default: reference book, click to edit your own positions)"):
    st.markdown("Edit the positions (add or delete rows). **An equity line is your stock via an index and a beta you choose** (no live prices: "
                "zero-cost app). Kinds: `bond` (10y, curve eur/us, optional spread), `future` (index sx5e / spx), `fxfwd` (USD sold forward), "
                "`option` (S&P put/call). Nominal in local currency, negative = short.")
    base = pd.DataFrame(P.CFG["lines"]).reindex(columns=COLS)
    base["spread"] = base["spread"].fillna("none")
    ed = st.data_editor(base, num_rows="dynamic", width="stretch", key="pf", column_config={
        "kind": st.column_config.SelectboxColumn(options=["bond", "future", "fxfwd", "option"], required=True),
        "ccy": st.column_config.SelectboxColumn(options=["EUR", "USD"], required=True),
        "curve": st.column_config.SelectboxColumn(options=["eur", "us"]),
        "spread": st.column_config.SelectboxColumn(options=["none", "eur_sov_spread", "baa_spread"]),
        "index": st.column_config.SelectboxColumn(options=["sx5e", "spx"]),
        "right": st.column_config.SelectboxColumn(options=["put", "call"]),
        "nominal": st.column_config.NumberColumn(format="%d")})
    nav = st.number_input("NAV (EUR), to express P&L in %", value=float(P.NAV), min_value=1.0, step=1e6)
try:
    lines = build(ed.reset_index(drop=True))
    if not lines:
        raise ValueError("add at least one line")
    pf = P.Portfolio(SNAP["state"], lines)
except (ValueError, TypeError, KeyError) as e:
    st.error(str(e))
    st.stop()
names = [L["name"] for L in lines]
lab_tab, crises, t1, t2, t3, t4, t5 = st.tabs(["Stress lab", "Crisis scenarios", "Ref: scenarios", "Ref: spot x vol", "Ref: cascade",
                                                  "Ref: reverse stress", "Ref: propagation backtest"])
with lab_tab:
    c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
    c0.selectbox("Start from a preset (illustrative, not a forecast)", list(lab.PRESETS), key="preset", on_change=apply_preset)
    regime = c1.selectbox("Risk regime", ["Normal", "Stressed"], help="Covariance used for the completion, VaR and ES: normal = EWMA on the full history, stressed = crisis windows only.")
    nu = None if c2.selectbox("Distribution", ["Gaussian", "Student-t (5 dof)"]) == "Gaussian" else 5
    complete = c3.toggle("Complete other factors", True, help="Factors you do not shock move to their conditional expectation given the ones you shock (e.g. equity down drags credit spreads up). Off: they stay at 0.")
    cols = st.columns(4)
    spec = {k: cols[i % 4].slider(lab.LABEL[k], *SLIDERS[k][:2], step=SLIDERS[k][2], key=f"s_{k}") for i, k in enumerate(SLIDERS)}
    spec = {k: v for k, v in spec.items() if v}
    if not spec:
        st.info("Move a slider or pick a preset: the lab revalues every line under your combined shock.")
    else:
        r = lab_cached(lines, tuple(spec.items()), regime, nu, complete)
        tot, b, a = r["pnl"].sum(), r["before"], r["after"]
        m = st.columns(5)
        m[0].metric("P&L (EUR m)", f"{tot / 1e6:.2f}", f"{tot / nav * 100:.2f} % NAV")
        m[1].metric("VaR 99 %, 10d, today (EUR m)", f"{b['var'] / 1e6:.2f}")
        m[2].metric("VaR 99 % after the shock", f"{a['var'] / 1e6:.2f}", f"{(a['var'] / b['var'] - 1) * 100:+.0f} %", delta_color="inverse")
        m[3].metric("ES 99 %, 10d, today (EUR m)", f"{b['es'] / 1e6:.2f}")
        m[4].metric("ES 99 % after the shock", f"{a['es'] / 1e6:.2f}", f"{(a['es'] / b['es'] - 1) * 100:+.0f} %", delta_color="inverse")
        st.caption(f"Mahalanobis d = {r['d']:.1f} (ordinal severity scale only: the Gaussian is rejected by the backtest, see the last tab)."
                   + (" **Alert**: the shocked block is ill-conditioned (cond > 1e6), the completion is unreliable." if r["cond"] > 1e6 else "")
                   + " VaR and ES after the shock = risk of the same book revalued from the shocked market (strikes, coupons and hedges unchanged, no rebalancing).")
        L1, L2 = st.columns(2)
        L1.markdown("**What we lost, by line (EUR m)**")
        L1.bar_chart(pd.Series(r["pnl"] / 1e6, index=r["names"]), horizontal=True)
        L2.markdown("**By risk driver, then what the combination adds (EUR m)**")
        L2.bar_chart((r["attr"] / 1e6).rename("EUR m"), horizontal=True)
        st.markdown("**Evolution as the shock builds (0 to 100 % of the combined shock)**")
        R = r["ramp"]
        E1, E2 = st.columns(2)
        E1.caption("P&L by line (EUR m)")
        E1.line_chart(R[r["names"]] / 1e6, x_label="shock severity, %")
        E2.caption("VaR and ES 99 % 10d measured from the shocked market (EUR m)")
        E2.line_chart(R[["VaR", "ES"]] / 1e6, x_label="shock severity, %")
        st.markdown("**Where the tail comes from: ES contribution by line, today vs after the shock (EUR m)**")
        st.bar_chart(pd.DataFrame({"today": b["es_by_line"] / 1e6, "after shock": a["es_by_line"] / 1e6}, index=r["names"]), horizontal=True)
    st.markdown("**Two-driver grid, total P&L in EUR m (other drivers stay at the slider values)**")
    g1, g2 = st.columns(2)
    k1 = g1.selectbox("Rows", list(SLIDERS), index=0, format_func=lab.LABEL.get)
    k2 = g2.selectbox("Columns", list(SLIDERS), index=1, format_func=lab.LABEL.get)
    if k1 != k2:
        S0 = pd.DataFrame(SNAP["cov" if regime == "Normal" else "cov_stressed"]).reindex(index=FACT, columns=FACT)
        G = lab.grid(pf, spec, k1, GRIDV[k1], k2, GRIDV[k2], S0, complete) / 1e6
        G.index, G.columns = [f"{v:+d}" for v in G.index], [f"{v:+d}" for v in G.columns]
        st.dataframe(G.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None), width="stretch")
    else:
        st.info("Pick two different drivers.")
with crises:
    X = np.array(list(SNAP["shocks"].values()))
    res = pd.DataFrame(pf.pnl(X), index=list(SNAP["shocks"]), columns=names) / 1e6
    tot = res.sum(axis=1)
    out = pd.DataFrame({"P&L (EUR m)": tot, "P&L (% NAV)": tot * 1e6 / nav * 100}).round(2)
    a, b = st.columns(2)
    a.dataframe(out, width="stretch")
    worst = out["P&L (% NAV)"].idxmin()
    b.metric("Worst scenario", worst, f"{out.loc[worst, 'P&L (% NAV)']:.2f} % NAV", delta_color="inverse")
    b.bar_chart(out["P&L (% NAV)"])
    pick2 = st.selectbox("Breakdown by line", list(res.index), index=list(res.index).index(worst))
    st.bar_chart(res.loc[pick2].rename("EUR m"), horizontal=True)
    st.caption("Five historical windows (observed factor moves) and four hypothetical scenarios (other factors completed by conditional "
               "Gaussian propagation). Full revaluation, instantaneous shocks, no theta.")
with t1:
    pick = st.selectbox("Scenario", sc.scenario)
    r = sc[sc.scenario == pick].iloc[0]
    a, b, c = st.columns(3)
    a.metric("P&L (% NAV)", f"{r.pnl_pct_nav:.2f}")
    b.metric("Mahalanobis d", "n/a (observed)" if pd.isna(r.mahalanobis_d) else f"{r.mahalanobis_d:.1f}")
    c.metric("Taylor-2 error (% NAV)", f"{r.taylor2_error_pct_nav:.2f}", "ALERT" if r.alert else None, delta_color="inverse")
    st.bar_chart(r.iloc[5:].astype(float).rename("EUR m"))
    st.dataframe(sc.iloc[:, :5].round(2), hide_index=True)
with t2:
    st.dataframe(pd.read_csv(OUTD / "spot_vol_grid.csv", index_col=0).style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None))
with t3:
    st.dataframe(pd.read_csv(OUTD / "cascade.csv", index_col=0))
with t4:
    rv = pd.read_csv(OUTD / "reverse.csv", index_col=0)
    st.line_chart(rv)
with t5:
    st.write("The Gaussian propagation is rejected by most PIT tests and its 90 % band over-covers: use d as an ordinal scale.")
    st.dataframe(pd.read_csv(OUTD / "backtest.csv", index_col=[0, 1]).round(3))
st.caption(f"Panel fingerprint {json.loads((OUTD / 'meta.json').read_text())['panel_fingerprint']}")
