"""
session_stats.py — Persistent per-session sweep/FBO statistics for NQ ("SweepStats" v1).

The compounding dataset behind the ICT/SMC stats product: for every trading
day it pins down session ranges and classifies what NY actually did at each
session level, then appends the record to a tracked JSON dataset. Re-runs
are idempotent (existing dates are kept unless --rebuild), so the dataset
grows day by day instead of being recomputed as a rolling snapshot like
results/amd_review.json.

Definitions (v1 — 5m bars, all times ET):
  Sessions
    Asia   : 6:00 PM – midnight, prior calendar day
    London : 2:00 AM – 5:00 AM
    NY     : 9:30 AM – 4:00 PM  (time buckets: 09:30-10:00, 10:00-11:00,
             11:00-13:00, 13:00-16:00)

  London manipulation
    "London swept Asia High/Low" = London range extends beyond the Asia
    extreme (range containment test, matching the AMD framing used in
    amd_session_review.py).

  First-touch classification (per level: Asia H/L, London H/L)
    The first NY 5m bar whose range breaches the level starts an episode.
    If that bar or either of the next 2 bars CLOSES back on the original
    side, the episode is a FAKEOUT (FBO) confirmed at that reclaim close;
    otherwise it is a BREAK. Levels never breached are NO-TOUCH.

  Follow-through
    Measured from the confirm close (reclaim close for fakeouts, breach
    close for breaks) in the direction the classification implies
    (fakeout = reversal, break = continuation):
      mfe_30 / mfe_60 / mfe_120 : max favorable move within 6/12/24 bars
      mae_120                   : max adverse move beyond the episode
                                  extreme within 24 bars (stop-run risk)

Run:
  python -m backtest.session_stats             # update dataset + print stats
  python -m backtest.session_stats --rebuild   # recompute all days from scratch
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from data.market_events import event_kinds

warnings.filterwarnings("ignore")

ET = ZoneInfo("America/New_York")

ASIA_START = time(18, 0)
LONDON_START = time(2, 0)
LONDON_END = time(5, 0)
NY_START = time(9, 30)
NY_END = time(16, 0)

CONFIRM_BARS = 2          # bars after the breach bar allowed to reclaim
FOLLOW_BARS = 24          # 2h of 5m bars for follow-through
DATASET_VERSION = 1       # bump if definitions change (invalidates old records)

DATASET_FILE = Path(__file__).parent / "session_stats_dataset.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TIME_BUCKETS = [
    ("09:30-10:00", time(9, 30), time(10, 0)),
    ("10:00-11:00", time(10, 0), time(11, 0)),
    ("11:00-13:00", time(11, 0), time(13, 0)),
    ("13:00-16:00", time(13, 0), time(16, 0)),
]


NQ_FRONT_CONID = 770561204  # Sep 2026 — update at quarterly roll


def _dl_5m_ibkr(conid: int = NQ_FRONT_CONID, period: str = "1m") -> pd.DataFrame:
    """
    5m bars from the IBKR CP Gateway HMDS endpoint (real CME feed, ~17k
    bars/call). Needs the gateway running and logged in; raises on any
    failure so the caller can fall back to yfinance. Only the front-month
    window is trustworthy — pre-roll far-month bars are ghost prints.
    """
    from data.ibkr import _SESSION, BASE_URL

    _SESSION.post(f"{BASE_URL}/hmds/auth/init", timeout=20)
    r = _SESSION.get(f"{BASE_URL}/hmds/history",
                     params={"conid": conid, "period": period, "bar": "5min",
                             "outsideRth": "true"}, timeout=120)
    r.raise_for_status()
    bars = [b for b in r.json().get("data", []) if "t" in b]
    if not bars:
        raise RuntimeError("HMDS returned no bars")
    df = pd.DataFrame(bars)
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df["t"], unit="ms", utc=True)).tz_convert(ET)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def _dl_5m_yf(ticker: str = "NQ=F") -> pd.DataFrame:
    df = yf.download(ticker, period="60d", interval="5m",
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    return df.sort_index()


def _dl_5m() -> tuple[pd.DataFrame, str]:
    """IBKR-first, yfinance fallback. Returns (bars, source_tag)."""
    try:
        df = _dl_5m_ibkr()
        return df, "ibkr"
    except Exception as e:
        print(f"    (IBKR unavailable: {type(e).__name__} — using yfinance)")
        return _dl_5m_yf(), "yfinance"


def _window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)]


def _session_hl(df: pd.DataFrame, day: date) -> dict:
    asia = _window(df,
                   datetime.combine(day - timedelta(days=1), ASIA_START, tzinfo=ET),
                   datetime.combine(day, time(0, 0), tzinfo=ET))
    london = _window(df,
                     datetime.combine(day, LONDON_START, tzinfo=ET),
                     datetime.combine(day, LONDON_END, tzinfo=ET))
    out = {}
    out["asia_high"] = round(float(asia["high"].max()), 2) if len(asia) >= 2 else None
    out["asia_low"] = round(float(asia["low"].min()), 2) if len(asia) >= 2 else None
    out["london_high"] = round(float(london["high"].max()), 2) if len(london) >= 2 else None
    out["london_low"] = round(float(london["low"].min()), 2) if len(london) >= 2 else None
    return out


def _bucket(ts: pd.Timestamp) -> str:
    t = ts.time()
    for name, lo, hi in TIME_BUCKETS:
        if lo <= t < hi:
            return name
    return "other"


def _first_touch(ny: pd.DataFrame, level: float, side: str) -> dict | None:
    """
    Classify the first NY interaction with a session level.
    side='high': level is a ceiling (breach = trade above).
    side='low' : level is a floor  (breach = trade below).
    """
    if side == "high":
        breached = ny["high"] > level
    else:
        breached = ny["low"] < level
    if not breached.any():
        return None

    i = int(breached.values.argmax())
    breach_ts = ny.index[i]

    # reclaim = a close back on the original side within the breach bar
    # or the next CONFIRM_BARS bars
    reclaim_j = None
    for j in range(i, min(i + 1 + CONFIRM_BARS, len(ny))):
        c = float(ny.iloc[j]["close"])
        if (side == "high" and c < level) or (side == "low" and c > level):
            reclaim_j = j
            break

    if reclaim_j is not None:
        kind = "fakeout"
        confirm_j = reclaim_j
        # fakeout reverses away from the level
        direction = "short" if side == "high" else "long"
    else:
        kind = "break"
        confirm_j = min(i + CONFIRM_BARS, len(ny) - 1)
        direction = "long" if side == "high" else "short"

    confirm_close = float(ny.iloc[confirm_j]["close"])

    # episode extreme = furthest breach up to the confirm bar
    seg = ny.iloc[i:confirm_j + 1]
    extreme = float(seg["high"].max()) if side == "high" else float(seg["low"].min())
    overshoot = abs(extreme - level)

    fwd = ny.iloc[confirm_j + 1: confirm_j + 1 + FOLLOW_BARS]

    def _mfe(bars: pd.DataFrame) -> float:
        if bars.empty:
            return 0.0
        if direction == "long":
            return max(0.0, float(bars["high"].max()) - confirm_close)
        return max(0.0, confirm_close - float(bars["low"].min()))

    if direction == "long":
        mae = max(0.0, extreme - float(fwd["low"].min())) if len(fwd) else 0.0
    else:
        mae = max(0.0, float(fwd["high"].max()) - extreme) if len(fwd) else 0.0

    return {
        "kind": kind,
        "direction": direction,
        "time": breach_ts.strftime("%H:%M"),
        "bucket": _bucket(breach_ts),
        "level": round(level, 2),
        "extreme": round(extreme, 2),
        "overshoot_pts": round(overshoot, 2),
        "confirm_close": round(confirm_close, 2),
        "mfe_30": round(_mfe(fwd.iloc[:6]), 2),
        "mfe_60": round(_mfe(fwd.iloc[:12]), 2),
        "mfe_120": round(_mfe(fwd), 2),
        "mae_120": round(mae, 2),
    }


def analyse_session(df: pd.DataFrame, day: date) -> dict | None:
    ny = _window(df,
                 datetime.combine(day, NY_START, tzinfo=ET),
                 datetime.combine(day, NY_END, tzinfo=ET))
    if len(ny) < 30:  # partial NY session — skip (holiday/short day)
        return None
    if ny.index[-1].time() < time(15, 30):  # early close (half-day) — skip
        return None

    hl = _session_hl(df, day)
    if hl["asia_high"] is None or hl["london_high"] is None:
        return None

    london_sweep = "none"
    swept_high = hl["london_high"] > hl["asia_high"]
    swept_low = hl["london_low"] < hl["asia_low"]
    if swept_high and swept_low:
        london_sweep = "both"
    elif swept_high:
        london_sweep = "asia_high"
    elif swept_low:
        london_sweep = "asia_low"

    touches = {}
    for name, level, side in [
        ("asia_high", hl["asia_high"], "high"),
        ("asia_low", hl["asia_low"], "low"),
        ("london_high", hl["london_high"], "high"),
        ("london_low", hl["london_low"], "low"),
    ]:
        touches[name] = _first_touch(ny, level, side)

    ny_open = float(ny.iloc[0]["open"])
    ny_close = float(ny.iloc[-1]["close"])

    return {
        "v": DATASET_VERSION,
        "date": day.isoformat(),
        "weekday": day.strftime("%a"),
        **hl,
        "london_sweep": london_sweep,
        "ny_open": round(ny_open, 2),
        "ny_high": round(float(ny["high"].max()), 2),
        "ny_low": round(float(ny["low"].min()), 2),
        "ny_close": round(ny_close, 2),
        "ny_direction": "up" if ny_close > ny_open else "down",
        "ny_change_pts": round(ny_close - ny_open, 2),
        "events": event_kinds(day),
        "first_touch": touches,
    }


# ── Aggregation ────────────────────────────────────────────────────────────────

def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    m = len(s) // 2
    return round(s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2, 2)


def aggregate(sessions: list[dict]) -> dict:
    sessions = [s for s in sessions if s.get("v") == DATASET_VERSION]

    # 1) London manipulation → NY direction
    london = {}
    for sweep in ("asia_high", "asia_low", "none", "both"):
        subset = [s for s in sessions if s["london_sweep"] == sweep]
        ups = sum(1 for s in subset if s["ny_direction"] == "up")
        london[sweep] = {
            "days": len(subset),
            "ny_up": ups,
            "ny_up_pct": _pct(ups, len(subset)),
            "median_ny_change_pts": _median([s["ny_change_pts"] for s in subset]),
        }

    # 2) First-touch outcomes per level
    levels = {}
    for lvl in ("asia_high", "asia_low", "london_high", "london_low"):
        eps = [s["first_touch"][lvl] for s in sessions if s["first_touch"].get(lvl)]
        n_days = len(sessions)
        fakes = [e for e in eps if e["kind"] == "fakeout"]
        breaks = [e for e in eps if e["kind"] == "break"]
        levels[lvl] = {
            "days_in_sample": n_days,
            "touched": len(eps),
            "touched_pct": _pct(len(eps), n_days),
            "fakeout": len(fakes),
            "break": len(breaks),
            "fakeout_pct": _pct(len(fakes), len(eps)),
            "fakeout_median_overshoot_pts": _median([e["overshoot_pts"] for e in fakes]),
            "fakeout_median_mfe60_pts": _median([e["mfe_60"] for e in fakes]),
            "fakeout_median_mfe120_pts": _median([e["mfe_120"] for e in fakes]),
            "fakeout_median_mae120_pts": _median([e["mae_120"] for e in fakes]),
            "fakeout_mfe60_ge30pts_pct": _pct(
                sum(1 for e in fakes if e["mfe_60"] >= 30), len(fakes)),
            "break_median_mfe60_pts": _median([e["mfe_60"] for e in breaks]),
        }

    # 3) Fakeout quality by time bucket (all levels pooled)
    buckets = {}
    all_eps = [e for s in sessions for e in s["first_touch"].values() if e]
    for name, _, _ in TIME_BUCKETS:
        eps = [e for e in all_eps if e["bucket"] == name]
        fakes = [e for e in eps if e["kind"] == "fakeout"]
        buckets[name] = {
            "touches": len(eps),
            "fakeout": len(fakes),
            "fakeout_pct": _pct(len(fakes), len(eps)),
            "fakeout_median_mfe60_pts": _median([e["mfe_60"] for e in fakes]),
        }

    # 4) Event-day behavior (OPEX/CPI/FOMC/... vs ordinary days) — flow days
    #    are regime overrides; publish their behavior only once n is honest
    def _ny_range(s: dict) -> float:
        return float(s["ny_high"]) - float(s["ny_low"])

    plain = [s for s in sessions if not s.get("events")]
    events = {"none": {
        "days": len(plain),
        "ny_up_pct": _pct(sum(1 for s in plain if s["ny_direction"] == "up"),
                          len(plain)),
        "median_ny_change_pts": _median([s["ny_change_pts"] for s in plain]),
        "median_ny_range_pts": _median([_ny_range(s) for s in plain]),
    }}
    kinds = sorted({k for s in sessions for k in (s.get("events") or [])})
    for kind in kinds:
        sub = [s for s in sessions if kind in (s.get("events") or [])]
        events[kind] = {
            "days": len(sub),
            "ny_up_pct": _pct(sum(1 for s in sub if s["ny_direction"] == "up"),
                              len(sub)),
            "median_ny_change_pts": _median([s["ny_change_pts"] for s in sub]),
            "median_ny_range_pts": _median([_ny_range(s) for s in sub]),
        }

    dates = sorted(s["date"] for s in sessions)
    return {
        "generated": datetime.now(ET).isoformat(timespec="seconds"),
        "dataset_version": DATASET_VERSION,
        "instrument": "NQ (CME E-mini, continuous front month)",
        "sample": {"sessions": len(sessions),
                   "from": dates[0] if dates else None,
                   "to": dates[-1] if dates else None},
        "london_manipulation": london,
        "first_touch": levels,
        "time_buckets": buckets,
        "event_days": events,
    }


# ── Persistence + main ─────────────────────────────────────────────────────────

def load_dataset() -> dict:
    if DATASET_FILE.exists():
        return json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    return {"meta": {"dataset_version": DATASET_VERSION, "instrument": "NQ=F"},
            "sessions": {}}


def main():
    rebuild = "--rebuild" in sys.argv
    print("\n=== SweepStats — NQ session dataset ===\n")

    ds = load_dataset()
    if rebuild or ds["meta"].get("dataset_version") != DATASET_VERSION:
        print("  (rebuilding dataset from scratch)")
        ds = {"meta": {"dataset_version": DATASET_VERSION, "instrument": "NQ=F"},
              "sessions": {}}

    print("  Downloading NQ 5m...")
    df, source = _dl_5m()
    if df.empty:
        print("ERROR: no data.")
        return
    print(f"    {len(df)} bars ({source}), "
          f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")

    # backfill event flags on records written before the events field existed
    # (pure function of the date — no bar data needed, safe for any session)
    backfilled = 0
    for key, rec in ds["sessions"].items():
        if "events" not in rec:
            rec["events"] = event_kinds(date.fromisoformat(key))
            backfilled += 1
    if backfilled:
        print(f"  Backfilled event flags on {backfilled} existing sessions")

    all_days = sorted({ts.date() for ts in df.index if ts.weekday() < 5})
    added = 0
    for day in all_days:
        key = day.isoformat()
        if key in ds["sessions"]:
            continue
        rec = analyse_session(df, day)
        if rec:
            rec["source"] = source
            ds["sessions"][key] = rec
            added += 1

    DATASET_FILE.write_text(
        json.dumps(ds, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  Dataset: {len(ds['sessions'])} sessions ({added} new) -> {DATASET_FILE.name}")

    summary = aggregate(list(ds["sessions"].values()))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "session_stats_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Summary -> {out}")

    # readable digest
    print(f"\n  Sample: {summary['sample']['sessions']} sessions "
          f"{summary['sample']['from']} -> {summary['sample']['to']}\n")
    print("  London manipulation -> NY direction:")
    for k, v in summary["london_manipulation"].items():
        if v["days"]:
            print(f"    London sweeps {k:<10} n={v['days']:>3}  "
                  f"NY up {v['ny_up_pct']}%  median NY {v['median_ny_change_pts']:+.1f}pts")
    print("\n  First NY touch of session levels:")
    for k, v in summary["first_touch"].items():
        if v["touched"]:
            print(f"    {k:<12} touched {v['touched']}/{v['days_in_sample']} days  "
                  f"fakeout {v['fakeout_pct']}%  "
                  f"median reversal 60m {v['fakeout_median_mfe60_pts']}pts  "
                  f"stop-run risk {v['fakeout_median_mae120_pts']}pts")
    print("\n  Fakeout rate by time of first touch:")
    for k, v in summary["time_buckets"].items():
        if v["touches"]:
            print(f"    {k}  touches={v['touches']:>3}  fakeout {v['fakeout_pct']}%  "
                  f"median reversal 60m {v['fakeout_median_mfe60_pts']}pts")
    print("\n  Event days vs ordinary days (NY behavior):")
    for k, v in summary["event_days"].items():
        if v["days"]:
            print(f"    {k:<14} n={v['days']:>3}  NY up {v['ny_up_pct']}%  "
                  f"median move {v['median_ny_change_pts']:+.1f}pts  "
                  f"median range {v['median_ny_range_pts']:.0f}pts")
    print()


if __name__ == "__main__":
    main()
