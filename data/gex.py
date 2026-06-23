"""
GEX Suite screenshot reader — captures the heatmap via Chrome DevTools Protocol.

Setup (before each session):
  1. Run scripts\launch_trading_chrome.bat  (opens Chrome on debug port 9222)
  2. Log in to GEX Suite in that Chrome window
  3. Navigate to the heatmap for the symbol you want (SPY / SPX / QQQ)
  4. Call screenshot_gex("SPY") from this module, or run:
       python data/gex.py

The screenshot is saved to results/ and can be read by Claude via vision.
"""

import json
import base64
import time
from pathlib import Path

import requests

CDP_URL = "http://localhost:9222"  # same Chrome instance as TradingView
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _get_tabs() -> list[dict]:
    try:
        return requests.get(f"{CDP_URL}/json", timeout=3).json()
    except requests.exceptions.ConnectionError:
        return []


def get_gex_tab() -> dict | None:
    """Return the first Chrome tab that has gexsuite.com open, or None."""
    tabs = _get_tabs()
    for t in tabs:
        if "gexsuite" in t.get("url", "").lower():
            return t
    return None


def screenshot_gex(symbol: str = "SPY") -> Path | None:
    """
    Screenshot the active GEX Suite tab.
    Returns the path to the PNG file so Claude can read it via vision.
    """
    import websocket

    tab = get_gex_tab()
    if tab is None:
        all_tabs = _get_tabs()
        if not all_tabs:
            print("[!] Chrome not running with debug port 9222.")
            print("    Run: scripts\\launch_trading_chrome.bat")
        else:
            print("[!] GEX Suite not open in debug Chrome.")
            print(f"    Open tabs: {[t.get('url','?')[:60] for t in all_tabs]}")
        return None

    ws_url = tab["webSocketDebuggerUrl"]
    try:
        ws = websocket.create_connection(ws_url, timeout=10)
        ws.send(json.dumps({
            "id": 1,
            "method": "Page.captureScreenshot",
            "params": {"format": "png", "quality": 95}
        }))
        result = json.loads(ws.recv())
        ws.close()
    except Exception as e:
        print(f"[!] CDP screenshot error: {e}")
        return None

    if "result" not in result or "data" not in result.get("result", {}):
        print(f"[!] Unexpected CDP response: {result}")
        return None

    img_bytes = base64.b64decode(result["result"]["data"])
    out_path  = RESULTS_DIR / f"gex_{symbol.lower()}.png"
    out_path.write_bytes(img_bytes)
    print(f"  GEX screenshot saved → {out_path.name}")
    return out_path


if __name__ == "__main__":
    tab = get_gex_tab()
    if tab:
        print(f"Found GEX tab: {tab['url'][:80]}")
        path = screenshot_gex("SPY")
        if path:
            print(f"Screenshot saved: {path}")
    else:
        print("No GEX Suite tab found — run scripts\\launch_trading_chrome.bat first.")
