"""
Fetch OHLCV data for NQ and ES from TradingView via tvdatafeed.
Saves CSVs to data/NQ/ and data/ES/ for each timeframe.

Timeframe purpose:
  15min  — AMD phase + market structure (FBOS, RBOS)
   5min  — Draw POIs (IDM orderblocks)
   3min  — Detect SMTs (2-candle OB traps)
   1min  — Entry execution (price returning to IDM POI)
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from tvDatafeed import TvDatafeed, Interval

# Patch signin headers so TradingView doesn't reject the login
from tvDatafeed.main import TvDatafeed as _TvDatafeed
_TvDatafeed._TvDatafeed__signin_headers = {
    "Referer": "https://www.tradingview.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.tradingview.com",
}

# ── credentials ──────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
USERNAME   = os.getenv("TV_USERNAME")
PASSWORD   = os.getenv("TV_PASSWORD")
AUTH_TOKEN = os.getenv("TV_AUTH_TOKEN")   # preferred — bypasses CAPTCHA

if not AUTH_TOKEN and (not USERNAME or not PASSWORD):
    raise SystemExit(
        "Set TV_AUTH_TOKEN in .env (preferred), or TV_USERNAME + TV_PASSWORD."
    )

# ── config ────────────────────────────────────────────────────────────────────
SYMBOLS = {
    "NQ": ("NQ1!", "CME_MINI"),
    "ES": ("ES1!", "CME_MINI"),
}

TIMEFRAMES = {
    "4h":    Interval.in_4_hour,
    "1h":    Interval.in_1_hour,
    "15min": Interval.in_15_minute,
    "5min":  Interval.in_5_minute,
    "3min":  Interval.in_3_minute,
    "1min":  Interval.in_1_minute,
}

BARS = {
    "4h":    5000,
    "1h":    5000,
    "15min": 5000,
    "5min":  15000,   # ~13k bars to reach Jan 26
    "3min":  20000,
    "1min":  10000,
}

DATA_DIR = Path(__file__).parent


def fetch_all():
    if AUTH_TOKEN:
        print("Authenticating via auth_token (no CAPTCHA)...")
        tv = TvDatafeed(auth_token=AUTH_TOKEN)
    else:
        print("Authenticating via username/password...")
        tv = TvDatafeed(username=USERNAME, password=PASSWORD)
    print("Authenticated.\n")

    for asset, (symbol, exchange) in SYMBOLS.items():
        for tf_name, interval in TIMEFRAMES.items():
            out_path = DATA_DIR / asset / f"{tf_name}.csv"
            print(f"Fetching {asset} {tf_name} ({symbol} @ {exchange}, {BARS[tf_name]} bars)...")

            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                n_bars=BARS[tf_name],
            )

            if df is None or df.empty:
                print(f"  WARNING: no data returned for {asset} {tf_name}")
                continue

            # Normalise columns: index → datetime, lowercase OHLCV
            df.index.name = "datetime"
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]]

            df.to_csv(out_path)
            print(f"  Saved {len(df)} rows → {out_path.relative_to(DATA_DIR.parent)}")

    print("\nDone.")


if __name__ == "__main__":
    fetch_all()
