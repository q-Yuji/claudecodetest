"""
tradeify_account.py — prop-account drawdown / eval guardrail.

CURRENT ACCOUNT (since 2026-07-17): MFFU (My Funded Futures) "Builder"
eval, acct MFFUEVBLDR401004033 — one of TWO Builder accounts bought
2026-07-16 (this file tracks the one being traded; the second is
untracked until the user asks for two-account support). All prior
Tradeify accounts (funded + evals) are blown; the historical Tradeify
rules live in git history.

Balances tracked relative to the account base, so base = $0.

MFFU Builder rules (per user, 2026-07-17):
  - Eval profit target    : +$3,000 total — NO consistency rule during
                            the eval; can pass in a single day.
  - EOD drawdown          : $2,000 (4%) below the highest EOD closing
                            balance — panel-confirmed off the MFFU stats
                            dashboard 2026-07-17 (floor $48,060.96 vs EOD
                            high $50,060.96). User quoted $2,100; the
                            dashboard says $2,000 — dashboard wins. No
                            intraday / daily loss limit reported.
  - Funded phase          : 50% consistency; max payout $2,000 per
                            request — a ~$4,000 balance is needed for the
                            max payout. Exact eligibility formula
                            unconfirmed; revisit when the eval passes.

State (highest EOD balance, today's starting balance, per-day profit
history) persists in tradeify_state.json next to this file.

Usage:
    from data.tradeify_account import check_guardrail, roll_day, record_reading

    roll_day(current_balance)          # call once at session start
    check_guardrail(current_balance)   # call any time for a live readout
"""

import json
from datetime import date
from pathlib import Path

STATE_FILE = Path(__file__).parent / "tradeify_state.json"

PHASE = "eval"  # MFFU Builder eval (acct MFFUEVBLDR401004033)

TRAILING_DRAWDOWN = 2000.0
DAILY_DRAWDOWN = None  # no daily loss limit reported for MFFU Builder
EVAL_TARGET = 3000.0
EVAL_CONSISTENCY_CAP = None  # Builder evals have no consistency rule
FUNDED_CONSISTENCY_CAP = 0.50
PAYOUT_CAP = 2000.0
PAYOUT_MIN_BALANCE = 4000.0  # balance needed for the max $2k payout

_DEFAULT_STATE = {
    "highest_eod_balance": 0.0,
    "day_start_balance": 0.0,
    "last_seen_balance": 0.0,
    "last_seen_date": None,
    "daily_profits": {},
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
    state.setdefault("daily_profits", {})

    if state["last_seen_date"] and state["last_seen_date"] != today:
        state["highest_eod_balance"] = max(
            state["highest_eod_balance"], state["last_seen_balance"]
        )
        # bank the completed day's P&L for the consistency rule
        closed_pnl = round(
            state["last_seen_balance"] - state["day_start_balance"], 2
        )
        state["daily_profits"][state["last_seen_date"]] = closed_pnl

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


def _eval_progress(current_balance: float, state: dict) -> dict:
    """Distance to the eval profit target. MFFU Builder evals have no
    consistency rule, so the target is a flat +$3,000 — passable in one day."""
    day_start = state["day_start_balance"]
    today_pnl = round(current_balance - day_start, 2)

    return {
        "today_pnl": today_pnl,
        "profit_needed_to_pass": round(max(EVAL_TARGET - current_balance, 0.0), 2),
        "required_total": EVAL_TARGET,
        "passed": current_balance >= EVAL_TARGET,
    }


def check_guardrail(current_balance: float) -> dict:
    """Compute floors and phase progress against the current balance."""
    state = record_reading(current_balance)

    trailing_floor = state["highest_eod_balance"] - TRAILING_DRAWDOWN

    result = {
        "phase": PHASE,
        "current_balance": current_balance,
        "trailing_floor": trailing_floor,
        "distance_to_trailing_floor": current_balance - trailing_floor,
    }

    # MFFU Builder has no intraday/daily loss limit in either phase.
    if PHASE == "funded" and DAILY_DRAWDOWN is not None:
        daily_floor = state["day_start_balance"] - DAILY_DRAWDOWN
        result["daily_floor"] = daily_floor
        result["distance_to_daily_floor"] = current_balance - daily_floor

    if PHASE == "eval":
        result["eval"] = _eval_progress(current_balance, state)
    else:
        # Funded payout math per user 2026-07-17: max $2k payout needs a
        # ~$4k balance. Exact eligibility formula unconfirmed — treated as
        # balance-above-(min_balance - cap), capped at PAYOUT_CAP.
        payout_available = max(
            0.0, min(current_balance - (PAYOUT_MIN_BALANCE - PAYOUT_CAP), PAYOUT_CAP)
        )
        result["payout_eligible"] = payout_available > 0
        result["payout_available"] = round(payout_available, 2)

    return result


def print_guardrail(current_balance: float):
    r = check_guardrail(current_balance)
    print(f"\n{'-'*46}")
    print(f"  MFFU Builder Buffer Check ({r['phase']})")
    print(f"{'-'*46}")
    print(f"  Current balance         : ${r['current_balance']:,.2f}")
    print(f"  Trailing EOD floor      : ${r['trailing_floor']:,.2f}  ({r['distance_to_trailing_floor']:+,.2f} cushion)")
    if "daily_floor" in r:
        print(f"  Daily floor             : ${r['daily_floor']:,.2f}  ({r['distance_to_daily_floor']:+,.2f} cushion)")
    if r["phase"] == "eval":
        e = r["eval"]
        if e["passed"]:
            print(f"  Eval                    : PASSED (total ${r['current_balance']:,.2f} >= ${e['required_total']:,.2f})")
        else:
            print(f"  Eval pass needs         : ${e['profit_needed_to_pass']:,.2f} more (target ${e['required_total']:,.2f}, no consistency rule)")
            print(f"  Today so far            : ${e['today_pnl']:+,.2f}")
    elif r["payout_eligible"]:
        print(f"  Payout available        : ${r['payout_available']:,.2f} (max ${PAYOUT_CAP:,.0f} at ${PAYOUT_MIN_BALANCE:,.0f}+)")
    else:
        print(f"  Payout eligible         : No")
    print(f"{'-'*46}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data/tradeify_account.py <current_balance>")
        raise SystemExit(1)

    print_guardrail(float(sys.argv[1]))
