"""Validation report (PDF) and demo GIF, both drawn from outputs/ only."""
import io
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

from .data import ROOT  # noqa: E402

OUTD = ROOT / "outputs"
SCOPE = """Scope. Synthetic 100 MEUR book, nine lines, ten observable factors, instantaneous shocks (no theta, no expiry).

Status. Self-review by the author: not an independent validation.

Method. Full revaluation (two-pillar zero curve, Black-Scholes with ATM vol = VIX) against delta / delta-gamma Taylor,
alert if the error exceeds 10 % of the P&L or 0.25 % of NAV. Partial shocks are completed by Gaussian conditional
propagation (EWMA 0.97 + Ledoit-Wolf covariance, square-root-of-time scaling); Mahalanobis d is an ordinal scale.

Checks performed. Fiche control cases for bond and put (reproduced to 0.01-0.02), put-call parity, third-order
slope of the Taylor remainder, QuantLib bond reference (gap 0.0098 % of nominal), asset-class cascade adds up,
rolling out-of-sample backtest of the propagation (non-overlapping 10-day periods, PIT, KS, Berkowitz, Wilson)."""


def _text_page(pdf, title, body):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold")
    fig.text(0.08, 0.90, "\n".join(textwrap.fill(p, 100) if "\n" not in p else p for p in body.split("\n\n")),
             fontsize=9, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def report():
    sc, bt = pd.read_csv(OUTD / "scenarios.csv"), pd.read_csv(OUTD / "backtest.csv", index_col=[0, 1])
    t5 = pd.read_csv(OUTD / "backtest_t5.csv", index_col=[0, 1])
    rg = pd.read_csv(OUTD / "regimes.csv")
    lq = json.loads((OUTD / "meta.json").read_text())["lqd"]
    frag = rg[rg.scenario == "fragmentation"].set_index("regime").pnl_pct_nav
    rv = pd.read_csv(OUTD / "reverse.csv")
    ks_rej = (bt.ks_p < 0.05).mean()
    findings = (f"Finding 1 (material). The Gaussian propagation is rejected by the KS test at 5 % for {ks_rej:.0%} of the "
                f"factor/pilot pairs ({len(bt)} tested, n = {int(bt.n.iloc[0])} periods each). The nominal 90 % band covers "
                f"{bt.cover90.min():.0%} to {bt.cover90.max():.0%}: conservative, with fat tails. Use d as an ordinal scale only.\n\n"
                "Finding 2. Reverse stress by bisection along the delta direction is not the exact optimum for a non-linear book "
                f"(d at 10 % loss: {rv.set_index('loss_pct_nav').loc[10, 'd_linear']:.1f} linear, "
                f"{rv.set_index('loss_pct_nav').loc[10, 'd_full']:.1f} full revaluation).\n\n"
                f"Finding 3. Taylor order 2 raises the alert on {int(sc.alert.sum())} of {len(sc)} scenarios: do not use sensitivities "
                "for the crisis-sized moves of the book (options and convexity).\n\n"
                f"Finding 4. A Student-t propagation (nu = 5, fixed in advance, not tuned) lowers the KS rejection rate from {ks_rej:.0%} "
                f"to {(t5.ks_p < 0.05).mean():.0%}: a partial fix, tails are not the only problem.\n\n"
                f"Finding 5. The result depends on the covariance regime: the fragmentation scenario costs {frag['full']:.1f} % of NAV "
                f"with the full sample, {frag['stressed']:.1f} % with stressed periods and {frag['post2021']:.1f} % since 2021.\n\n"
                f"Finding 6. The US corporate basket line is modelled with the BAA10Y spread; LQD regressed on it implies a duration of "
                f"{lq['implied_duration']:.1f} (R2 {lq['r2']:.2f}, 10-day tracking error {lq['tracking_error_10d_pct']:.1f} %): large basis risk.\n\n"
                "Opinion. Fit for use as an exploratory stress tool with the conditions above; not fit to produce a probability.")
    with PdfPages(OUTD / "validation_report.pdf") as pdf:
        _text_page(pdf, "stress-engine: validation report (self-review)", SCOPE + "\n\n" + findings)
        fig, ax = plt.subplots(1, 2, figsize=(11.69, 5))
        sc.set_index("scenario").pnl_pct_nav.plot.barh(ax=ax[0], title="P&L, % NAV")
        ax[1].plot(rv.loss_pct_nav, rv.d_linear, label="linear")
        ax[1].plot(rv.loss_pct_nav, rv.d_full, label="full revaluation")
        ax[1].set(title="Reverse stress: d*(L*)", xlabel="loss, % NAV", ylabel="Mahalanobis d")
        ax[1].legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def gif():
    sc = pd.read_csv(OUTD / "scenarios.csv")
    cols = sc.columns[5:]
    frames = []
    for _, r in sc.iterrows():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        r[cols].astype(float).plot.barh(ax=ax, color="#2a6f97")
        ax.set(title=f"{r.scenario}: {r.pnl_pct_nav:+.2f} % NAV", xlabel="EUR m", xlim=(-6, 6))
        fig.tight_layout()
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=80)
        plt.close(fig)
        frames.append(Image.open(b).convert("P"))
    (ROOT / "docs").mkdir(exist_ok=True)
    frames[0].save(ROOT / "docs" / "demo.gif", save_all=True, append_images=frames[1:], duration=1500, loop=0)


if __name__ == "__main__":
    report()
    gif()
