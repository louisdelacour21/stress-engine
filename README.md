# stress-engine

Combined multi-factor stress tests on a synthetic 100 MEUR portfolio (9 lines, 10 market factors).
**Status: self-review, one author, synthetic portfolio, not an independent validation.**

* Ten factors: EUR 2y/10y AAA (ECB), EUR sovereign spread, US 2y/10y (FRED), BAA10Y, VIX, EURUSD, Euro Stoxx 50, S&P 500.
* Shocks are instantaneous (no theta, no expiry). Full revaluation (bonds on a two-pillar zero curve, Black-Scholes options with ATM vol = VIX) versus delta / delta-gamma Taylor, with an alert when the error exceeds 10 % of the P&L or 0.25 % of NAV.
* Partial hypothetical shocks are completed by Gaussian conditional propagation (EWMA 0.97 + Ledoit-Wolf covariance, square-root-of-time scaling). The Mahalanobis distance d is a relative plausibility scale, not a return period.
* Reverse stress: smallest-d move losing L % of NAV (closed form on deltas, then full-revaluation bisection along the same direction, which is not the exact optimum for non-linear books).
* Rolling out-of-sample backtest of the propagation (424 non-overlapping 10-day periods from 2008).

## Findings, as measured
The Gaussian propagation is **rejected** by the PIT tests (KS p < 0.01 for most factors, pilot Euro Stoxx 50 or US 10y) and its 90 % band covers 92-98 %, i.e. it is conservative with fat tails. Treat d as ordinal only.
Crisis shocks are recomputed from the data (they differ from rough textbook figures). Raw series are not redistributed (Yahoo / Cboe / ICE licences); only derived outputs in `outputs/`.

## Run
```
pip install -e ".[data,dev]"
make reproduce      # ~20 s, writes outputs/
pytest
```
Not done: Student copula, Shapley attribution, stressed ES, EBA mapping, Streamlit demo, GIF, validation report.
MIT licence.
