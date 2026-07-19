"""
market_events.py — flow/event-day classification for any session date.

Why: some days are driven by option-expiry and macro flows, not market
structure (user, 2026-07-19: the 07-17 intraday V-recovery was OPEX flow;
CPI usually distributes one outsized candle). These flags feed three
places: the morning brief calendar, the Situation Room's Today's Script
caveat chips, and the session dataset — so the stats engine can eventually
publish TESTED claims about event days instead of folklore.

Rule-computed (no feed needed):
  opex          monthly OPEX — third Friday of every month; the largest
                option open interest expires, pinning gamma rolls off
                through the session (charm flows run all day, and the
                post-OPEX session trades "unclamped")
  quad_witching third Friday of Mar/Jun/Sep/Dec — index futures + options
                expire together (emitted alongside opex)
  vix_exp       VIX futures/options settlement — the Wednesday 30 days
                before the NEXT month's third Friday (holiday shifts are
                rare and not modeled)

Pinned from official schedules (rule-proof, needed for dataset backfill;
extend each December):
  cpi           BLS CPI release, 08:30 ET (bls.gov schedule)
  fomc          FOMC rate decision, 14:00 ET statement / 14:30 presser
                (federalreserve.gov calendar)

NFP is deliberately absent: "first Friday" has real exceptions and a wrong
flag stamped into the dataset is worse than none — add it as a pinned
schedule if wanted.

Usage:
  from data.market_events import events_for
  events_for(date(2026, 7, 17))
  python -m data.market_events [YYYY-MM-DD]
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

# BLS 2026 schedule (all 08:30 ET) — release dates, not reference months.
CPI_DATES = {
    date(2026, 1, 13), date(2026, 2, 13), date(2026, 3, 11),
    date(2026, 4, 10), date(2026, 5, 12), date(2026, 6, 10),
    date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 11),
    date(2026, 10, 14), date(2026, 11, 10), date(2026, 12, 10),
}

# Federal Reserve 2026 calendar — decision (second) day of each meeting.
FOMC_DATES = {
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
}

_SCHEDULE_YEARS = {2026}  # years the pinned CPI/FOMC schedules cover


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)  # first Friday
    return d + timedelta(days=14)


def vix_expiration(year: int, month: int) -> date:
    """VIX settlement in the given month: 30 days before the next month's
    third Friday (a Wednesday by construction)."""
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return third_friday(ny, nm) - timedelta(days=30)


def events_for(day: date) -> list[dict]:
    """All flow/event flags for one date, most market-moving first."""
    out = []
    if day in FOMC_DATES:
        out.append({"kind": "fomc", "label": "FOMC DECISION 14:00",
                    "note": "rate decision 14:00 ET, presser 14:30 — "
                            "afternoon repricing risk"})
    if day in CPI_DATES:
        out.append({"kind": "cpi", "label": "CPI 08:30",
                    "note": "one outsized candle at the print is the norm"})
    if day == third_friday(day.year, day.month):
        if day.month in (3, 6, 9, 12):
            out.append({"kind": "quad_witching", "label": "QUAD WITCHING",
                        "note": "futures + options expire together — "
                                "heaviest expiry flows of the quarter"})
        out.append({"kind": "opex", "label": "OPEX FRIDAY",
                    "note": "monthly expiry — charm/pin flows can dominate "
                            "structure ALL session, not one candle"})
    if day == vix_expiration(day.year, day.month):
        out.append({"kind": "vix_exp", "label": "VIX EXPIRATION",
                    "note": "VIX settles this morning — vol-complex flows "
                            "into the AM print"})
    return out


def event_kinds(day: date) -> list[str]:
    return [e["kind"] for e in events_for(day)]


def schedule_covers(day: date) -> bool:
    """False if CPI/FOMC pins don't cover this year (rule events still work)."""
    return day.year in _SCHEDULE_YEARS


if __name__ == "__main__":
    d = (date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1
         else date.today())
    evts = events_for(d)
    if not schedule_covers(d):
        print(f"WARNING: {d.year} CPI/FOMC schedules not pinned — "
              "only rule-based events shown")
    if not evts:
        print(f"{d}: no flow events")
    for e in evts:
        print(f"{d}: [{e['kind']}] {e['label']} — {e['note']}")
