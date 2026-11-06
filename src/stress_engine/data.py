"""Loaders (ECB SDMX, FRED CSV, Yahoo) with a local cache, and the common-calendar level panel of the ten factors."""
import hashlib
import io
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FACTORS = list(yaml.safe_load((ROOT / "config" / "factors.yaml").read_text(encoding="utf-8")))


def _cached(name: str, fetch) -> pd.Series:
    path = RAW / f"{name}.csv"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        fetch().rename(name).to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return s.dropna()


def ecb(key: str) -> pd.Series:
    url = f"https://data-api.ecb.europa.eu/service/data/{key}?startPeriod=2004-01-01"
    r = requests.get(url, headers={"Accept": "text/csv"}, timeout=60)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    return pd.Series(d["OBS_VALUE"].values, index=pd.to_datetime(d["TIME_PERIOD"]))


def fred(key: str) -> pd.Series:
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={key}", timeout=60)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")


def yahoo(ticker: str) -> pd.Series:
    import yfinance as yf
    h = yf.download(ticker, start="2004-01-01", progress=False, auto_adjust=False)["Close"]
    s = h.iloc[:, 0] if isinstance(h, pd.DataFrame) else h
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def level(name: str, spec: dict) -> pd.Series:
    """Level series of one factor (percent for rates and spreads)."""
    src = spec["source"]
    if src == "ecb_spread":
        return _cached(name, lambda: ecb(spec["all"]) - ecb(spec["aaa"]))
    return _cached(name, lambda: {"ecb": ecb, "fred": fred, "yahoo": yahoo}[src](spec["key"]))


def panel() -> pd.DataFrame:
    """Ten factor levels on the days where all ten are observed (inner join: no filled values)."""
    spec = yaml.safe_load((ROOT / "config" / "factors.yaml").read_text(encoding="utf-8"))
    df = pd.concat({n: level(n, s) for n, s in spec.items()}, axis=1).dropna()
    return df


def lqd() -> pd.Series:
    return _cached("lqd", lambda: yahoo("LQD"))


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def changes(df: pd.DataFrame | None = None, h: int = 1) -> pd.DataFrame:
    """h-day factor changes: basis points for rates and spreads, log-changes otherwise."""
    import numpy as np
    spec = yaml.safe_load((ROOT / "config" / "factors.yaml").read_text(encoding="utf-8"))
    df = panel() if df is None else df
    out = {}
    for n, s in spec.items():
        out[n] = df[n].diff(h) * 100 if s["kind"] == "bp" else np.log(df[n]).diff(h)
    return pd.DataFrame(out).dropna()
