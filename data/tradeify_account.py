"""
tradeify_account.py — Track Tradeify funded-account drawdown/payout rules.

Rules (funded phase, no consistency rule):
  - Trailing EOD drawdown : $2,000 below the highest EOD closing balance
                            (floor never below -$2,000, since funded balance
                            starts at $0)
  - Daily drawdown        : $1,000 below today's starting balance
  - Payout eligibility    : balance >= $2,100
  - Payout available      : min(balance - 2100, 1000), if eligible

State (highest EOD balance, today's starting balance) persists in
tradeify_state.json next to this file, since the trailing floor and daily
floor depend on history, not just the current balance.

Usage:
    from data.tradeify_account import check_guardrail, roll_day, record_reading

    roll_day(current_balance)          # call once at session start
    check_guardrail(current_balance)   # call any time for a live readout
"""

import json
from datetime import date
from pathlib import Path

STATE_FILE = Path(__file__).parent / "tradeify_state.json"

TRAILING_DRAWDOWN = 2000.0
DAILY_DRAWDOWN = 1000.0
PAYOUT_BUFFER = 2100.0
PAYOUT_CAP = 1000.0

_DEFAULT_STATE = {
    "highest_eod_balance": 0.0,
    "day_start_balance": 0.0,
    "last_seen_balance": 0.0,
    "last_seen_date": None,
}


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return dict(_DEFAULT_STATE)


def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def roll_day(current_balance: float, today: str | None = None) -> dict:
    """
    Call once at the start of a trading session. If the date has changed
    since the last reading, banks yesterday's last-seen balance as an EOD
    close and starts a fresh daily floor from the current balance.
    """
    today = today or date.today().isoformat()
    state = _load_state()

    if state["last_seen_date"] and state["last_seen_date"] != today:
        state["highest_eod_balance"] = max(
            state["highest_eod_balance"], state["last_seen_balance"]
        )

    state["day_start_balance"] = current_balance
    state["last_seen_balance"] = current_balance
    state["last_seen_date"] = today
    _save_state(state)
    return state


def record_reading(current_balance: float, today: str | None = None) -> dict:
    """Update the latest-seen balance without resetting the daily floor (for on-demand checks mid-session)."""
    today = today or date.today().isoformat()
    state = _load_state()

    if not state["last_seen_date"]:
        return roll_day(current_balance, today)

    if state["last_seen_date"] != today:
        return roll_day(current_balance, today)

    state["last_seen_balance"] = current_balance
    _save_state(state)
    return state


def check_guardrail(current_balance: float) -> dict:
    """Compute floors and payout eligibility against the current balance."""
    state = record_reading(current_balance)

    trailing_floor = state["highest_eod_balance"] - TRAILING_DRAWDOWN
    daily_floor = state["day_start_balance"] - DAILY_DRAWDOWN

    payout_eligible = current_balance >= PAYOUT_BUFFER
    payout_available = (
        min(current_balance - PAYOUT_BUFFER, PAYOUT_CAP) if payout_eligible else 0.0
    )

    return {
        "current_balance": current_balance,
        "trailing_floor": trailing_floor,
        "distance_to_trailing_floor": current_balance - trailing_floor,
        "daily_floor": daily_floor,
        "distance_to_daily_floor": current_balance - daily_floor,
        "payout_eligible": payout_eligible,
        "payout_available": round(payout_available, 2),
    }


def print_guardrail(current_balance: float):
    r = check_guardrail(current_balance)
    print(f"\n{'-'*46}")
    print(f"  Tradeify Buffer Check")
    print(f"{'-'*46}")
    print(f"  Current balance         : ${r['current_balance']:,.2f}")
    print(f"  Trailing EOD floor      : ${r['trailing_floor']:,.2f}  ({r['distance_to_trailing_floor']:+,.2f} cushion)")
    print(f"  Daily floor             : ${r['daily_floor']:,.2f}  ({r['distance_to_daily_floor']:+,.2f} cushion)")
    if r["payout_eligible"]:
        print(f"  Payout available        : ${r['payout_available']:,.2f}")
    else:
        print(f"  Payout eligible         : No (${PAYOUT_BUFFER - r['current_balance']:,.2f} short of ${PAYOUT_BUFFER:,.0f} buffer)")
    print(f"{'-'*46}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data/tradeify_account.py <current_balance>")
        raise SystemExit(1)

    print_guardrail(float(sys.argv[1]))
