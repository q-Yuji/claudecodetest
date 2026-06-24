"""
Trading session startup — single command to launch everything.

Usage:
    python start_session.py

What it does:
    1. Launches Chrome with TradingView + GEX Suite + IBKR Gateway
    2. Starts the AI chat server (localhost:8765)
    3. Generates the morning brief dashboard and opens it in the browser

After running: log in to IBKR (first tab), TradingView, and GEX Suite,
then tell Claude Code "session ready" to begin chart monitoring.
"""

import subprocess
import sys
import socket
import time
from pathlib import Path

BASE = Path(__file__).parent


def _port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except OSError:
        return False


def main():
    print("=" * 55)
    print("  Jarvis — Trading Session Startup")
    print("=" * 55)

    # 1. Launch Chrome + IBKR
    bat = BASE / "scripts" / "launch_trading_chrome.bat"
    if bat.exists():
        print("\n[1/3] Launching Chrome (TV + GEX Suite) and IBKR Gateway...")
        subprocess.Popen(
            ["cmd", "/c", str(bat)],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        print("      Waiting 6s for Chrome to open...")
        time.sleep(6)
    else:
        print("[1/3] WARNING: scripts/launch_trading_chrome.bat not found — skipping.")

    # 2. Morning brief (also starts chat server internally)
    print("\n[2/3] Generating morning brief...")
    subprocess.run([sys.executable, str(BASE / "morning_brief.py")])

    # 3. Confirm chat server
    print("\n[3/3] Checking AI chat server...")
    if _port_open(8765):
        print("      Chat server running on http://localhost:8765")
    else:
        print("      Starting chat server...")
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen([sys.executable, str(BASE / "chat_server.py")], creationflags=flags)

    print("\n" + "=" * 55)
    print("  Done. Next steps:")
    print("  1. Log in to IBKR at https://localhost:5000 (tab 1)")
    print("  2. Log in to TradingView (tab 2)")
    print("  3. Log in to GEX Suite (tab 3)")
    print("  4. Tell Claude Code: 'session ready'")
    print("=" * 55)


if __name__ == "__main__":
    main()
