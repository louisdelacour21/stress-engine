"""Scenario shocks: the five historical windows recomputed from the series, and the hypothetical sets."""
import numpy as np
import pandas as pd
import yaml

from .data import ROOT, panel

CFG = yaml.safe_load((ROOT / "config" / "scenarios.yaml").read_text(encoding="utf-8"))
SPEC = yaml.safe_load((ROOT / "config" / "factors.yaml").read_text(encoding="utf-8"))


def _asof(df: pd.DataFrame, day: str) -> pd.Series:
    return df.loc[:day].iloc[-1]


def historical(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Factor move over each window (bp or log-change), one row per crisis."""
    df = panel() if df is None else df
    rows = {}
    for name, w in CFG["historical"].items():
        a, b = _asof(df, w["start"]), _asof(df, w["end"])
        rows[name] = {n: (b[n] - a[n]) * 100 if s["kind"] == "bp" else float(np.log(b[n] / a[n])) for n, s in SPEC.items()}
    return pd.DataFrame(rows).T


def hypothetical() -> dict[str, dict[str, float]]:
    return CFG["hypothetical"]
