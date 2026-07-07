"""
tradeify_account.py — Track Tradeify account drawdown and eval/funded rules.

Current phase: EVAL ($50k account; balances tracked relative to the $50k
base, so base = $0).

Eval pass rules (PHASE = "eval"):
  - Profit target         : +$3,000 total
  - 40% consistency rule  : no single day's profit may exceed 40% of total
                            profit at the time of passing. Overshooting a day
                            doesn't fail — it raises the required total to
                            best_day / 0.40.
  - No payouts during eval.

Drawdown rules:
  - Trailing EOD drawdown : $2,000 below the highest EOD closing balance
                            (floor never below -$2,000) — both phases
  - Daily drawdown        : $1,000 below today's starting balance —
                            FUNDED PHASE ONLY (evals have no daily loss limit)

Funded-phase payout rules (dormant until PHASE = "funded"):
  - Payout eligibility    : balance >= $2,100
  - Payout available      : min(balance - 2100, 1000), if eligible

State (highest EOD balance, today's starting balance, per-day profit
history for the consistency rule) persists in tradeify_state.json next to
this file, since the floors and consistency math depend on history, not
just the current balance.

Usage:
    from data.tradeify_account import check_guardrail, roll_day, record_reading

    roll_day(current_balance)          # call once at session start
    check_guardrail(current_balance)   # call any time for a live readout
"""

import json
from datetime import date
from pathlib import Path

STATE_FILE = Path(__file__).parent / "tradeify_state.json"

PHASE = "funded"  # eval passed 2026-07-07

TRAILING_DRAWDOWN = 2000.0
DAILY_DRAWDOWN = 1000.0
EVAL_TARGET = 3000.0
CONSISTENCY_CAP = 0.40
PAYOUT_BUFFER = 2100.0
PAYOUT_CAP = 1000.0

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
    """Distance to the eval profit target, with 40% consistency math."""
    day_start = state["day_start_balance"]
    today_pnl = round(current_balance - day_start, 2)
    past_days = dict(state.get("daily_profits", {}))
    best_day = max(list(past_days.values()) + [today_pnl, 0.0])

    # The consistency rule can push the effective target above $3,000:
    # every day (including the best one) must be <= 40% of the final total.
    required_total = max(EVAL_TARGET, best_day / CONSISTENCY_CAP)

    # Largest P&L today can reach while still allowing a clean pass at the
    # $3,000 target: t <= CAP * (day_start + t)  =>  t <= CAP*day_start/(1-CAP)
    today_clean_cap = round(CONSISTENCY_CAP * day_start / (1 - CONSISTENCY_CAP), 2)

    return {
        "today_pnl": today_pnl,
        "best_day_profit": round(best_day, 2),
        "profit_needed_to_pass": round(max(required_total - current_balance, 0.0), 2),
        "required_total": round(required_total, 2),
        "today_consistency_cap": today_clean_cap,
        "passed": current_balance >= required_total,
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

    # Daily loss limit only exists on funded accounts; evals have none.
    if PHASE == "funded":
        daily_floor = state["day_start_balance"] - DAILY_DRAWDOWN
        result["daily_floor"] = daily_floor
        result["distance_to_daily_floor"] = current_balance - daily_floor

    if PHASE == "eval":
        result["eval"] = _eval_progress(current_balance, state)
    else:
        payout_eligible = current_balance >= PAYOUT_BUFFER
        payout_available = (
            min(current_balance - PAYOUT_BUFFER, PAYOUT_CAP)
            if payout_eligible
            else 0.0
        )
        result["payout_eligible"] = payout_eligible
        result["payout_available"] = round(payout_available, 2)

    return result


def print_guardrail(current_balance: float):
    r = check_guardrail(current_balance)
    print(f"\n{'-'*46}")
    print(f"  Tradeify Buffer Check ({r['phase']})")
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
            print(f"  Eval pass needs         : ${e['profit_needed_to_pass']:,.2f} more (target ${e['required_total']:,.2f})")
            print(f"  Today so far            : ${e['today_pnl']:+,.2f} (clean-pass cap ${e['today_consistency_cap']:,.2f})")
    elif r["payout_eligible"]:
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
