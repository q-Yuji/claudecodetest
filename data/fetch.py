"""
Fetch OHLCV data for NQ and ES using yfinance.
Saves CSVs to data/NQ/ and data/ES/ for each timeframe.

Timeframe purpose:
  4h    — long-term trend bias
  1h    — trend bias (4H+1H must agree)
  15min — macro structure context
   5min — FBOS + RBOS detection, IDM POI formation
   3min — SMT detection (resampled from 1min)
   1min — OTE entry execution

yfinance availability:
  5min / 15min : last 60 days  (~Jan 22 from today)
  1h           : last 730 days
  4h           : resampled from 1h
  3min         : resampled from 1min
  1min         : last 7 days
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

SYMBOLS = {
    "NQ": "NQ=F",
    "ES": "ES=F",
}

DATA_DIR = Path(__file__).parent


def _save(df: pd.DataFrame, asset: str, tf: str):
    out = DATA_DIR / asset / f"{tf}.csv"
    df.index.name = "datetime"
    df.to_csv(out)
    print(f"  Saved {len(df)} rows → {out.relative_to(DATA_DIR.parent)}  "
          f"({df.index[0].date()} → {df.index[-1].date()})")


def _fetch(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    # yfinance returns MultiIndex columns when auto_adjust=True in some versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe."""
    return df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()


def fetch_all():
    for asset, ticker in SYMBOLS.items():
        print(f"\n── {asset} ({ticker}) ──────────────────────────────")

        # 1h — 730 days (basis for 4h resample)
        df_1h = _fetch(ticker, "1h", "730d")
        if df_1h.empty:
            print(f"  WARNING: no 1h data for {asset}")
            continue
        _save(df_1h, asset, "1h")

        # 4h — resampled from 1h
        df_4h = _resample(df_1h, "4h")
        _save(df_4h, asset, "4h")

        # 5min — 60 days (~Jan 22)
        df_5 = _fetch(ticker, "5m", "60d")
        if not df_5.empty:
            _save(df_5, asset, "5min")
        else:
            print(f"  WARNING: no 5min data for {asset}")

        # 15min — 60 days
        df_15 = _fetch(ticker, "15m", "60d")
        if not df_15.empty:
            _save(df_15, asset, "15min")

        # 1min — 7 days
        df_1m = _fetch(ticker, "1m", "7d")
        if not df_1m.empty:
            _save(df_1m, asset, "1min")

            # 3min — resampled from 1min
            df_3 = _resample(df_1m, "3min")
            _save(df_3, asset, "3min")
        else:
            print(f"  WARNING: no 1min data for {asset}")

    print("\nDone.")


if __name__ == "__main__":
    fetch_all()
