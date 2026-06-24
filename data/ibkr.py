"""
IBKR Client Portal Gateway — REST client for positions, P&L, and options data.

One-time setup:
  1. Download the gateway from:
     https://www.interactivebrokers.com/en/trading/ibkr-api-gl.php
     → "Client Portal API Gateway" → clientportal.gw.zip
  2. Unzip to C:/ibkr-gateway/
  3. Before each trading session, run:
       cd C:/ibkr-gateway
       bin/run.bat root/conf.yaml
  4. Open https://localhost:5000 in Chrome and log in with your IBKR credentials
  5. Keep that terminal open while trading

Usage:
  python data/ibkr.py
"""

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://localhost:5000/v1/api"
_SESSION = requests.Session()
_SESSION.verify = False  # Gateway uses a self-signed certificate


def _get(endpoint: str, **params) -> dict | list:
    r = _SESSION.get(f"{BASE_URL}{endpoint}", params=params or None, timeout=10)
    r.raise_for_status()
    return r.json()


def get_accounts() -> list[dict]:
    """Return list of accounts (usually just one)."""
    return _get("/portfolio/accounts")


def get_positions(account_id: str) -> list[dict]:
    """Return open positions with symbol, size, market value, unrealised P&L."""
    return _get(f"/portfolio/{account_id}/positions/0")


def get_account_summary(account_id: str) -> dict:
    """Return account summary: daily P&L, net liquidation value, buying power."""
    return _get(f"/portfolio/{account_id}/summary")


def get_market_data_snapshot(conids: list[int]) -> list[dict]:
    """
    Live price snapshot for a list of contract IDs.
    Fields: 31=last, 84=bid, 85=ask, 87=close, 88=open
    """
    return _get(
        "/iserver/marketdata/snapshot",
        conids=",".join(str(c) for c in conids),
        fields="31,84,85,87,88"
    )


def print_positions_summary(account_id: str):
    positions = get_positions(account_id)
    summary   = get_account_summary(account_id)

    print(f"\n{'-'*58}")
    print(f"  IBKR POSITIONS  --  {account_id}")
    print(f"{'-'*58}")

    if not positions:
        print("  No open positions.")
    else:
        print(f"  {'Contract':<42} {'Qty':>5}  {'Mkt Val':>10}  {'P&L':>10}")
        print(f"  {'-'*42}  {'-'*5}  {'-'*10}  {'-'*10}")
        for p in positions:
            sym  = (p.get("contractDesc") or p.get("ticker") or "?")[:42]
            qty  = p.get("position", 0)
            val  = p.get("mktValue", 0) or 0
            pnl  = p.get("unrealizedPnl", 0) or 0
            print(f"  {sym:<42} {qty:>+5.0f}  ${val:>9,.0f}  ${pnl:>+9,.0f}")

    nl   = (summary.get("netliquidation") or {}).get("amount", 0)
    dpnl = (summary.get("dailypnl")       or {}).get("amount", 0)
    bp   = (summary.get("buyingpower")    or {}).get("amount", 0)

    print(f"\n  Net Liquidation : ${nl:,.2f}")
    print(f"  Daily P&L       : ${dpnl:+,.2f}")
    print(f"  Buying Power    : ${bp:,.2f}")
    print(f"{'-'*58}\n")


if __name__ == "__main__":
    try:
        accounts = get_accounts()
    except requests.exceptions.ConnectionError:
        print("\n[!] Cannot reach IBKR Client Portal Gateway.")
        print("    Make sure it is running:")
        print("      cd C:\\ibkr-gateway")
        print("      bin\\run.bat root\\conf.yaml")
        print("    Then log in at: https://localhost:5000\n")
        raise SystemExit(1)

    if not accounts:
        print("No accounts returned -- are you logged in at https://localhost:5000 ?")
        raise SystemExit(1)

    account_id = accounts[0]["accountId"]
    print(f"Connected -- account: {account_id}")
    print_positions_summary(account_id)
