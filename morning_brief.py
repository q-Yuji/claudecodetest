"""
Morning Brief — pre-session market analysis.

Run before market open:
  python morning_brief.py

Pulls:
  - Overnight price context for NQ / SPY / QQQ / VIX (yfinance)
  - Economic calendar for today — high-impact USD events (ForexFactory)
  - Market news headlines (Yahoo Finance RSS)
  - GEX Suite heatmap screenshot (Chrome CDP on port 9222)
  - IBKR positions and Greeks (Client Portal Gateway on localhost:5000)

Output: console brief + results/morning_brief.json
"""

import json
import xml.etree.ElementTree as ET
from datetime import datetime, date
from pathlib import Path

import yfinance as yf
import requests

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Overnight price context ────────────────────────────────────────────────────

def get_overnight_context() -> dict:
    """Pull yesterday close → current price for NQ futures, SPY, QQQ, VIX."""
    symbols = {"NQ": "NQ=F", "SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX"}
    ctx = {}
    for name, ticker in symbols.items():
        try:
            data = yf.download(ticker, period="2d", interval="1d",
                               auto_adjust=True, progress=False)
            if len(data) >= 2:
                close      = data["Close"].squeeze()
                prev_close = float(close.iloc[-2])
                last       = float(close.iloc[-1])
                chg_pct    = (last - prev_close) / prev_close * 100
                ctx[name] = {
                    "prev_close": round(prev_close, 2),
                    "last":       round(last, 2),
                    "chg_pct":    round(chg_pct, 2),
                }
        except Exception as e:
            ctx[name] = {"error": str(e)}
    return ctx


# ── Economic calendar ──────────────────────────────────────────────────────────

def get_econ_calendar() -> list[dict]:
    """Today's high/medium-impact USD events from ForexFactory."""
    try:
        resp = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=5,
            headers={**_HEADERS, "Referer": "https://www.forexfactory.com/"},
        )
        try:
            raw = resp.json()
        except Exception:
            return [{"error": "Calendar feed blocked (try again during market hours)"}]
        today = date.today().isoformat()
        events = []
        for e in raw:
            if e.get("country") != "USD":
                continue
            if e.get("impact") not in ("High", "Medium"):
                continue
            if e.get("date", "")[:10] != today:
                continue
            events.append({
                "time":     e["date"][11:16],
                "title":    e.get("title", ""),
                "impact":   e.get("impact", ""),
                "forecast": e.get("forecast", "") or "-",
                "previous": e.get("previous", "") or "-",
            })
        return sorted(events, key=lambda x: x["time"])
    except Exception as e:
        return [{"error": str(e)}]


# ── News headlines ─────────────────────────────────────────────────────────────

def get_news_headlines(n: int = 6) -> list[str]:
    """Top Nasdaq/NQ-related headlines from Yahoo Finance RSS."""
    try:
        resp = requests.get(
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US",
            timeout=5, headers=_HEADERS,
        )
        root  = ET.fromstring(resp.text)
        items = root.findall(".//item")
        return [
            item.findtext("title", "").strip()
            for item in items[:n]
            if item.findtext("title", "").strip()
        ]
    except Exception as e:
        return [f"error: {e}"]


# ── IBKR positions ─────────────────────────────────────────────────────────────

def get_ibkr_data() -> dict:
    """Pull IBKR positions and account summary. Returns empty dict if unavailable."""
    try:
        from data.ibkr import get_accounts, get_positions, get_account_summary
        accounts = get_accounts()
        if not accounts:
            return {"error": "No accounts — log in at https://localhost:5000"}
        acc_id    = accounts[0]["accountId"]
        positions = get_positions(acc_id)
        summary   = get_account_summary(acc_id)

        nl   = (summary.get("netliquidation") or {}).get("amount", 0)
        dpnl = (summary.get("dailypnl")       or {}).get("amount", 0)
        bp   = (summary.get("buyingpower")    or {}).get("amount", 0)

        return {
            "account":    acc_id,
            "net_liq":    nl,
            "daily_pnl":  dpnl,
            "buying_pow": bp,
            "positions":  positions,
        }
    except requests.exceptions.ConnectionError:
        return {"error": "IBKR Gateway not running - start bin\\run.bat root\\conf.yaml"}
    except Exception as e:
        return {"error": str(e)}


# ── GEX Suite screenshot ────────────────────────────────────────────────────────

def capture_gex_charts() -> dict[str, str | None]:
    """Screenshot GEX Suite. Returns {symbol: path_or_None}."""
    try:
        from data.gex import screenshot_gex
        path = screenshot_gex("SPY")
        return {"SPY": str(path) if path else None}
    except Exception as e:
        return {"error": str(e)}


# ── Print and save ──────────────────────────────────────────────────────────────

def print_brief(ctx: dict, calendar: list, news: list, ibkr: dict, gex: dict):
    now = datetime.now().strftime("%a %d %b %Y  %H:%M")
    print(f"\n{'='*60}")
    print(f"  MORNING BRIEF  --  {now}")
    print(f"{'='*60}")

    # Overnight
    print("\n  OVERNIGHT CONTEXT")
    for sym, d in ctx.items():
        if "error" in d:
            print(f"    {sym:<4}  error: {d['error']}")
        else:
            arrow = "^" if d["chg_pct"] >= 0 else "v"
            print(f"    {sym:<4}  {arrow} {d['chg_pct']:+.2f}%   last: {d['last']}")

    # Economic calendar
    print("\n  TODAY'S ECONOMIC EVENTS (USD)")
    if not calendar:
        print("    No high-impact USD events today.")
    elif "error" in (calendar[0] if calendar else {}):
        print(f"    [{calendar[0]['error']}]")
    else:
        for e in calendar:
            impact_tag = "[HIGH]" if e["impact"] == "High" else "[MED] "
            print(f"    {e['time']}  {impact_tag}  {e['title']:<40}  "
                  f"fcst: {e['forecast']}  prev: {e['previous']}")

    # News
    print("\n  MARKET NEWS")
    for headline in news:
        if headline.startswith("error:"):
            print(f"    [{headline}]")
        else:
            print(f"    - {headline}")

    # IBKR
    print("\n  IBKR POSITIONS")
    if "error" in ibkr:
        print(f"    [{ibkr['error']}]")
    else:
        print(f"    Account: {ibkr['account']}   Net Liq: ${ibkr['net_liq']:,.0f}   "
              f"Daily P&L: ${ibkr['daily_pnl']:+,.0f}")
        positions = ibkr.get("positions") or []
        if not positions:
            print("    No open positions.")
        else:
            for p in positions:
                sym  = (p.get("contractDesc") or p.get("ticker") or "?")[:45]
                qty  = p.get("position", 0)
                pnl  = p.get("unrealizedPnl", 0) or 0
                print(f"    {sym:<45}  qty: {qty:+.0f}  P&L: ${pnl:+,.0f}")

    # GEX
    print("\n  GEX SUITE")
    if "error" in gex:
        print(f"    [{gex['error']}]")
    else:
        for sym, path in gex.items():
            print(f"    {sym}: {'saved → ' + path if path else 'not captured'}")

    print(f"\n{'='*60}\n")


def main():
    print("  Fetching market data...")
    ctx      = get_overnight_context()
    calendar = get_econ_calendar()
    news     = get_news_headlines()
    ibkr     = get_ibkr_data()
    gex      = capture_gex_charts()

    print_brief(ctx, calendar, news, ibkr, gex)

    output = {
        "generated": datetime.now().isoformat(),
        "overnight": ctx,
        "calendar":  calendar,
        "news":      news,
        "ibkr":      {k: v for k, v in ibkr.items() if k != "positions"},
        "positions": ibkr.get("positions", []),
        "gex_paths": gex,
    }
    out_path = RESULTS_DIR / "morning_brief.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Saved -> {out_path.name}")


if __name__ == "__main__":
    main()
