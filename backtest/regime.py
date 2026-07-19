"""
regime.py — daily trend/chop regime classification for NQ.

Why (user, 2026-07-19): playbooks are regime-gated. Their STDV projection
strategy "works in consolidation, dies in trends" (March–May 2026 trending
tape killed it; mid-July chop revived it) — and fade/fakeout stats
plausibly share that gate. This pins a falsifiable, price-only regime
definition so the claim can be TESTED instead of eyeballed, and so the
Situation Room can show today's regime next to the flow-event chips.

Definition (v1, daily closes, walk-forward safe):
  Kaufman efficiency ratio over the last ER_WINDOW=20 completed sessions:
      ER = |close[t] − close[t−20]| / Σ|close[i] − close[i−1]|
  ER = 1 means price went somewhere in a straight line; ER = 0 means all
  movement cancelled out.
    ER ≥ 0.35 → "trend_up" / "trend_down" (by sign of the net move)
    ER ≤ 0.20 → "chop"
    else      → "mixed"
  A session's regime uses only closes BEFORE its date — what a trader
  could have known at 9:28 that morning.

Run:
  python -m backtest.regime            # classify today + recent history
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd
import yfinance as yf

ER_WINDOW = 20
TREND_MIN = 0.35
CHOP_MAX = 0.20


def classify_closes(closes: list[float]) -> dict | None:
    """Regime from a list of daily closes (the last ER_WINDOW+1 are used).
    Returns {"regime", "er", "net_pts"} or None if not enough history."""
    if len(closes) < ER_WINDOW + 1:
        return None
    window = closes[-(ER_WINDOW + 1):]
    net = window[-1] - window[0]
    path = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
    er = abs(net) / path if path else 0.0
    if er >= TREND_MIN:
        regime = "trend_up" if net > 0 else "trend_down"
    elif er <= CHOP_MAX:
        regime = "chop"
    else:
        regime = "mixed"
    return {"regime": regime, "er": round(er, 3), "net_pts": round(net, 2)}


def daily_closes(ticker: str = "NQ=F", period: str = "1y") -> pd.Series:
    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"] if "Close" in df.columns else df["close"]
    close.index = pd.to_datetime(close.index).date
    return close.dropna()


def classify_map(closes: pd.Series) -> dict[str, dict]:
    """date-iso -> regime dict for every date, each computed walk-forward
    (only closes strictly before that date are used)."""
    dates = list(closes.index)
    vals = [float(v) for v in closes.values]
    out = {}
    for i, d in enumerate(dates):
        r = classify_closes(vals[:i])  # strictly before day d
        if r:
            out[d.isoformat()] = r
    return out


def current_regime(closes: pd.Series | None = None) -> dict | None:
    """Regime as of NOW (all completed closes) — for 'today' displays."""
    if closes is None:
        closes = daily_closes()
    return classify_closes([float(v) for v in closes.values])


def main() -> None:
    closes = daily_closes()
    if closes.empty:
        print("ERROR: no daily data")
        return
    cur = current_regime(closes)
    print(f"NQ regime as of {closes.index[-1]}: {cur['regime'].upper()} "
          f"(ER {cur['er']}, net {cur['net_pts']:+.0f} pts over {ER_WINDOW} sessions)")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cmap = classify_map(closes)
    print(f"\nlast {n} sessions:")
    prev = None
    for d in list(cmap)[-n:]:
        r = cmap[d]
        flip = "  <- flip" if prev and r["regime"] != prev else ""
        print(f"  {d}  {r['regime']:<10} ER {r['er']:.2f}{flip}")
        prev = r["regime"]


if __name__ == "__main__":
    main()
