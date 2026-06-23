"""
Morning Brief — pre-session market analysis.

Run before market open:
  python morning_brief.py

Pulls:
  - Overnight price context for SPY / QQQ / VIX (yfinance)
  - GEX Suite heatmap screenshot (Chrome CDP on port 9223)
  - IBKR positions and Greeks (Client Portal Gateway on localhost:5000)

Output: console brief + results/morning_brief.json
"""

import json
from datetime import datetime, date
from pathlib import Path

import yfinance as yf
import requests

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Overnight price context ────────────────────────────────────────────────────

def get_overnight_context() -> dict:
    """Pull yesterday close → current pre-market price for SPY, QQQ, VIX."""
    symbols = {"SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX"}
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


# ── IBKR positions ─────────────────────────────────────────────────────────────

def get_ibkr_data() -> dict:
    """Pull IBKR positions and account summary. Returns empty dict if unavailable."""
    try:
        from data.ibkr import get_accounts, get_positions, get_account_summary
        accounts = get_accounts()
        if not accounts:
            return {"error": "No accounts — log in at https://localhost:5000"}
        acc_id   = accounts[0]["accountId"]
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
        return {"error": "IBKR Gateway not running — start bin\\run.bat root\\conf.yaml"}
    except Exception as e:
        return {"error": str(e)}


# ── GEX Suite screenshot ────────────────────────────────────────────────────────

def capture_gex_charts() -> dict[str, str | None]:
    """Screenshot GEX Suite for SPY and QQQ. Returns {symbol: path_or_None}."""
    try:
        from data.gex import screenshot_gex
        return {
            "SPY": str(screenshot_gex("SPY")) if screenshot_gex("SPY") else None,
            "QQQ": str(screenshot_gex("QQQ")) if screenshot_gex("QQQ") else None,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Print and save ──────────────────────────────────────────────────────────────

def print_brief(ctx: dict, ibkr: dict, gex: dict):
    now = datetime.now().strftime("%a %d %b %Y  %H:%M")
    print(f"\n{'='*58}")
    print(f"  MORNING BRIEF  —  {now}")
    print(f"{'='*58}")

    # Overnight
    print("\n  OVERNIGHT CONTEXT")
    for sym, d in ctx.items():
        if "error" in d:
            print(f"    {sym:<4}  error: {d['error']}")
        else:
            arrow = "▲" if d["chg_pct"] >= 0 else "▼"
            print(f"    {sym:<4}  {arrow} {d['chg_pct']:+.2f}%   last: {d['last']}")

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
    print("\n  GEX SUITE SCREENSHOTS")
    if "error" in gex:
        print(f"    [{gex['error']}]")
    else:
        for sym, path in gex.items():
            status = path if path else "not captured"
            print(f"    {sym}: {status}")

    print(f"\n{'='*58}\n")


def main():
    print("  Fetching market data...")
    ctx  = get_overnight_context()
    ibkr = get_ibkr_data()
    gex  = capture_gex_charts()

    print_brief(ctx, ibkr, gex)

    output = {
        "generated":  datetime.now().isoformat(),
        "overnight":  ctx,
        "ibkr":       {k: v for k, v in ibkr.items() if k != "positions"},
        "positions":  ibkr.get("positions", []),
        "gex_paths":  gex,
    }
    out_path = RESULTS_DIR / "morning_brief.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Saved → {out_path.name}")


if __name__ == "__main__":
    main()
