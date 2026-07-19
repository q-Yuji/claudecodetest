"""scoreboard.py — walk-forward grading of Today's Script (roadmap feature 5).

"The morning call that grades itself." For every session in
backtest/session_stats_dataset.json, reconstruct what Today's Script
would have said that morning using ONLY the sessions before it (the same
london_manipulation bucket math as session_stats.aggregate), then grade
that call against what New York actually did. Every grade is
out-of-sample by construction, and the whole record is reproducible from
the append-only dataset — the dataset's git history IS the tamper-proof
archive, so no separate card archive is needed.

Two things get graded:
  1. The directional lean — the bucket's ny_up_pct at the time. A call is
     only issued once the bucket has >= MIN_DAYS prior days and the lean
     isn't a coin flip (exactly 50%); other days are NO CALL.
  2. The headline fakeout claim — every first touch on a call-eligible day
     is an out-of-sample trial of the fakeout rate the page was claiming
     that morning (pooled prior touches, >= MIN_TOUCHES).

Deliberately stdlib-only: situation_room.py imports compute_from_file()
to render The Record panel without dragging in the pipeline's pandas
stack.

Run:
  python -m backtest.scoreboard          # print record, write results/scoreboard.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

DATASET_FILE = Path(__file__).parent / "session_stats_dataset.json"
OUT_FILE = Path(__file__).parent.parent / "results" / "scoreboard.json"

DATASET_VERSION = 1
MIN_DAYS = 5      # bucket days needed before a directional call is issued
MIN_TOUCHES = 20  # pooled prior touches needed before the fakeout claim counts

SWEEPS = ("asia_high", "asia_low", "none", "both")


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _grade_day(s: dict, prior: list[dict]) -> dict:
    """One walk-forward row: the call the Script would have made, graded."""
    sweep = s["london_sweep"]
    subset = [p for p in prior if p["london_sweep"] == sweep]
    n = len(subset)
    ups = sum(1 for p in subset if p["ny_direction"] == "up")
    up_pct = _pct(ups, n)

    row = {
        "date": s["date"],
        "weekday": s["weekday"],
        "pattern": sweep,
        "prior_days": n,
        "said_up_pct": up_pct,
        "actual_direction": s["ny_direction"],
        "actual_change_pts": s["ny_change_pts"],
        "call": None,
        "hit": None,
    }
    if n < MIN_DAYS:
        row["no_call_reason"] = "low sample"
    elif up_pct == 50.0:
        row["no_call_reason"] = "coin flip"
    else:
        row["call"] = "up" if up_pct > 50.0 else "down"
        row["hit"] = row["call"] == s["ny_direction"]
    return row


def compute_scoreboard(dataset: dict) -> dict:
    sessions = sorted(
        (s for s in (dataset.get("sessions") or {}).values()
         if isinstance(s, dict) and s.get("v") == DATASET_VERSION),
        key=lambda s: s["date"])

    days: list[dict] = []
    oos_touches = oos_fakeouts = 0
    claim_sum = 0.0
    prior_touches = prior_fakeouts = 0

    for i, s in enumerate(sessions):
        row = _grade_day(s, sessions[:i])

        # grade the headline fakeout claim on days it was actually claimable
        todays = [e for e in (s.get("first_touch") or {}).values() if e]
        if prior_touches >= MIN_TOUCHES and todays:
            claim = 100.0 * prior_fakeouts / prior_touches
            fakes = sum(1 for e in todays if e["kind"] == "fakeout")
            oos_touches += len(todays)
            oos_fakeouts += fakes
            claim_sum += claim * len(todays)
            row["fakeout_claim_pct"] = round(claim, 1)
            row["fakeout_touches"] = len(todays)
            row["fakeout_fakeouts"] = fakes
        prior_touches += len(todays)
        prior_fakeouts += sum(1 for e in todays if e["kind"] == "fakeout")

        days.append(row)

    graded = [d for d in days if d["hit"] is not None]
    hits = sum(1 for d in graded if d["hit"])

    streak = 0
    for d in reversed(graded):
        if streak == 0:
            streak = 1 if d["hit"] else -1
        elif (streak > 0) == d["hit"]:
            streak += 1 if d["hit"] else -1
        else:
            break

    by_pattern = {}
    for sweep in SWEEPS:
        sub = [d for d in graded if d["pattern"] == sweep]
        h = sum(1 for d in sub if d["hit"])
        by_pattern[sweep] = {"calls": len(sub), "hits": h,
                             "hit_pct": _pct(h, len(sub))}

    return {
        "generated": datetime.now(ET).isoformat(timespec="seconds"),
        "dataset_version": DATASET_VERSION,
        "method": {
            "min_days": MIN_DAYS,
            "min_touches": MIN_TOUCHES,
            "note": "walk-forward: each call uses only sessions before its "
                    "date; graded against ny_direction (5m closes)",
        },
        "record": {
            "calls": len(graded),
            "hits": hits,
            "hit_pct": _pct(hits, len(graded)),
            "first_call": graded[0]["date"] if graded else None,
            "last_call": graded[-1]["date"] if graded else None,
            "streak": streak,
            "last10": [bool(d["hit"]) for d in graded[-10:]],
            "no_call_days": len(days) - len(graded),
        },
        "by_pattern": by_pattern,
        "fakeout_oos": {
            "touches": oos_touches,
            "fakeouts": oos_fakeouts,
            "oos_pct": _pct(oos_fakeouts, oos_touches),
            "avg_claim_pct": round(claim_sum / oos_touches, 1) if oos_touches else None,
        },
        "latest": graded[-1] if graded else None,
        "days": days,
    }


def compute_from_file(path: Path = DATASET_FILE) -> dict:
    return compute_scoreboard(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    sb = compute_from_file()
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(sb, indent=1), encoding="utf-8")

    r = sb["record"]
    print(f"scoreboard -> {OUT_FILE}")
    print(f"  directional record : {r['hits']}/{r['calls']} hits "
          f"({r['hit_pct']}%)  streak {r['streak']:+d}  "
          f"[{r['first_call']} -> {r['last_call']}]")
    print(f"  no-call days       : {r['no_call_days']}")
    fo = sb["fakeout_oos"]
    print(f"  fakeout claim OOS  : claimed ~{fo['avg_claim_pct']}%, "
          f"ran {fo['oos_pct']}% over {fo['touches']} touches")
    for k, v in sb["by_pattern"].items():
        if v["calls"]:
            print(f"    {k:<10} {v['hits']}/{v['calls']} ({v['hit_pct']}%)")
    lt = sb["latest"]
    if lt:
        print(f"  latest graded call : {lt['date']} {lt['pattern']} said "
              f"{lt['call']} {lt['said_up_pct']}% -> NY {lt['actual_direction']} "
              f"{lt['actual_change_pts']:+.1f} pts -> "
              f"{'HIT' if lt['hit'] else 'MISS'}")


if __name__ == "__main__":
    main()
