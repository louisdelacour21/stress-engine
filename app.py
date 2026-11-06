"""Streamlit demo on the precomputed outputs/ (no live computation, no raw market data)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

O = Path(__file__).parent / "outputs"
st.set_page_config(page_title="Stress engine", layout="wide")
st.title("Stress engine")
st.caption("Self-review, synthetic 100 MEUR portfolio, instantaneous shocks. Precomputed outputs only.")
sc = pd.read_csv(O / "scenarios.csv")
t1, t2, t3, t4, t5 = st.tabs(["Scenarios", "Spot x vol", "Cascade", "Reverse stress", "Propagation backtest"])
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
    st.dataframe(pd.read_csv(O / "spot_vol_grid.csv", index_col=0).style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None))
with t3:
    st.dataframe(pd.read_csv(O / "cascade.csv", index_col=0))
with t4:
    rv = pd.read_csv(O / "reverse.csv", index_col=0)
    st.line_chart(rv)
with t5:
    st.write("The Gaussian propagation is rejected by most PIT tests and its 90 % band over-covers: use d as an ordinal scale.")
    st.dataframe(pd.read_csv(O / "backtest.csv", index_col=[0, 1]).round(3))
st.caption(f"Panel fingerprint {json.loads((O / 'meta.json').read_text())['panel_fingerprint']}")
