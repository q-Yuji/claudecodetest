"""
append_gex_history.py — Bank today's live GexSuite levels into the tracked history.

Merges results/gex_levels.json (written by Claude during session-startup
step 3 from the live pine labels — a Python script cannot call the
TradingView MCP itself) into backtest/gex_history.json, keyed by date.
Run every session morning: it's the only chance to capture that day's NQ
gamma levels before they age out of the indicator's ~2-week lookback.

Idempotent: re-running on the same day replaces that day's entry. Never
deletes prior dates. Refuses stale input (gex_levels.json has held
weeks-old levels before) unless --force-date is passed.

Run:
  python -m backtest.append_gex_history [--force-date]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "gex_history.json"
LEVELS_FILE = Path(__file__).parent.parent / "results" / "gex_levels.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="append_gex_history")
    parser.add_argument("--force-date", action="store_true",
                        help="append even if gex_levels.json is not from today")
    parser.add_argument("--levels-file", type=Path, default=LEVELS_FILE,
                        help="capture file to merge (override for testing)")
    parser.add_argument("--history-file", type=Path, default=HISTORY_FILE,
                        help="history file to merge into (override for testing)")
    args = parser.parse_args(argv)

    capture = json.loads(args.levels_file.read_text(encoding="utf-8"))
    cap_date = capture["timestamp"][:10]
    if cap_date != date.today().isoformat() and not args.force_date:
        print(f"gex_levels.json is stale ({cap_date}) — recapture step 3 first")
        return 1

    history = json.loads(args.history_file.read_text(encoding="utf-8"))
    history[cap_date] = {
        "source": "live_capture",
        "symbol": capture["symbol"],
        "levels": [{"name": l["name"], "price": l["price"]}
                   for l in capture["levels"]],
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }

    # match the existing file byte-for-byte on untouched entries:
    # indent=2, ensure_ascii=False, no trailing newline
    args.history_file.write_text(
        json.dumps({d: history[d] for d in sorted(history)},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(f"banked {cap_date}: {len(capture['levels'])} levels "
          f"({len(history)} dates in history)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
