"""
import_history.py — Backfill the SweepStats dataset from a purchased vendor file.

Ingests ~12 months of NQ intraday bars (Databento / FirstRateData one-off
CSV, see backtest/vendor_data/README.md) and appends session records to
backtest/session_stats_dataset.json using the SAME classification code path
as the live engine (backtest.session_stats.analyse_session — definitions
are not forked). Every backfilled record is tagged "source": "vendor".

Rules:
  - Append/backfill only: a date already in the dataset (live source) is
    NEVER rewritten. Overlap days are classified anyway and compared to the
    live record, printing MATCH/MISMATCH per level — a timezone or
    double-conversion bug shows up here immediately.
  - Vendor timestamps are localized per --tz (default utc) then converted
    to ET exactly once.
  - 1m files are resampled to 5m (label=left, closed=left — the yfinance
    convention the engine was built on).
  - Per-contract files (a symbol/contract column) are handled by picking,
    for each session-day window (prior day 18:00 ET → 16:00 ET), the
    contract with the most volume in that window; no record spans a roll.

Run:
  python -m backtest.import_history <csv_path> [--tz utc|et] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

from backtest.session_stats import (
    DATASET_FILE, DATASET_VERSION, ET, analyse_session,
)

OHLCV = ["open", "high", "low", "close", "volume"]
LEVELS = ("asia_high", "asia_low", "london_high", "london_low")


def load_vendor_csv(path: Path, tz: str) -> pd.DataFrame:
    """Vendor CSV -> tz-aware ET 5m OHLCV frame (contract column kept if present)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
    elif "ts_event" in df.columns:  # Databento naming
        ts = pd.to_datetime(df["ts_event"])
    elif "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        raise SystemExit(
            "unrecognized CSV header — need timestamp,o,h,l,c,v or date,time,o,h,l,c,v")

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC" if tz == "utc" else ET)
    elif tz == "et":
        print("  note: file already tz-aware; --tz et ignored (no double-convert)")

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    sym_col = next((c for c in ("symbol", "contract", "ticker") if c in df.columns), None)
    keep = OHLCV + ([sym_col] if sym_col else [])
    out = df[keep].copy()
    out[OHLCV] = out[OHLCV].astype(float)
    out.index = pd.DatetimeIndex(ts).tz_convert(ET)
    return out.sort_index(), sym_col


def _ensure_5m(df: pd.DataFrame) -> pd.DataFrame:
    step = df.index.to_series().diff().dropna().median()
    if step == pd.Timedelta(minutes=5):
        return df
    if step == pd.Timedelta(minutes=1):
        return (df.resample("5min", label="left", closed="left")
                  .agg({"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"})
                  .dropna())
    raise SystemExit(f"unsupported bar interval: {step} (need 1m or 5m)")


def build_session_frame(df: pd.DataFrame, sym_col: str | None) -> pd.DataFrame:
    """Continuous-enough 5m frame: per session-day, bars from that day's
    highest-volume contract only. Single-series files pass through."""
    if sym_col is None:
        return _ensure_5m(df)

    days = sorted({ts.date() for ts in df.index if ts.weekday() < 5})
    pieces = []
    for day in days:
        w0 = datetime.combine(day - timedelta(days=1), time(18, 0), tzinfo=ET)
        w1 = datetime.combine(day, time(16, 0), tzinfo=ET)
        window = df[(df.index >= w0) & (df.index < w1)]
        if window.empty:
            continue
        front = window.groupby(sym_col)["volume"].sum().idxmax()
        pieces.append(window[window[sym_col] == front][OHLCV])
    if not pieces:
        raise SystemExit("no session windows found in file")
    merged = pd.concat(pieces)
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    return _ensure_5m(merged)


def compare_overlap(live: dict, vendor: dict) -> list[str]:
    lines = []
    for lvl in LEVELS:
        a, b = live["first_touch"].get(lvl), vendor["first_touch"].get(lvl)
        ka = a["kind"] if a else "no-touch"
        kb = b["kind"] if b else "no-touch"
        tag = "MATCH" if ka == kb else "MISMATCH"
        lines.append(f"    {tag}  {lvl:<12} live={ka:<8} vendor={kb}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="import_history",
        description="Backfill SweepStats from a purchased NQ intraday CSV "
                    "(see backtest/vendor_data/README.md).")
    parser.add_argument("csv_path", type=Path, help="vendor CSV file")
    parser.add_argument("--tz", choices=("utc", "et"), default="utc",
                        help="timezone of naive timestamps in the file (default utc)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be appended; write nothing")
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE,
                        help="dataset file override (testing)")
    args = parser.parse_args(argv)

    raw, sym_col = load_vendor_csv(args.csv_path, args.tz)
    df = build_session_frame(raw, sym_col)
    print(f"  {len(df)} 5m bars, {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}"
          + (f" (per-contract file, column '{sym_col}')" if sym_col else ""))

    ds = json.loads(args.dataset.read_text(encoding="utf-8"))
    if ds["meta"].get("dataset_version") != DATASET_VERSION:
        raise SystemExit("dataset version mismatch — refusing to backfill")

    added, overlaps, mismatches = [], 0, 0
    for day in sorted({ts.date() for ts in df.index if ts.weekday() < 5}):
        rec = analyse_session(df, day)
        if rec is None:
            continue
        key = day.isoformat()
        if key in ds["sessions"]:
            overlaps += 1
            print(f"  overlap {key} — keeping live record:")
            lines = compare_overlap(ds["sessions"][key], rec)
            mismatches += sum(1 for l in lines if "MISMATCH" in l)
            print("\n".join(lines))
        else:
            rec["source"] = "vendor"
            added.append((key, rec))

    if args.dry_run:
        for key, _ in added:
            print(f"  would append {key} (vendor)")
        print(f"  DRY RUN: {len(added)} would be appended, {overlaps} overlaps "
              f"({mismatches} level mismatches), nothing written")
        return 0

    for key, rec in added:
        ds["sessions"][key] = rec
    ds["sessions"] = {k: ds["sessions"][k] for k in sorted(ds["sessions"])}
    args.dataset.write_text(
        json.dumps(ds, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  appended {len(added)} vendor sessions "
          f"({overlaps} overlaps kept live, {mismatches} level mismatches) "
          f"-> {args.dataset.name} now {len(ds['sessions'])} sessions")
    if added:
        print("  dataset grew — commit backtest/session_stats_dataset.json "
              "and rerun python -m backtest.daily_stats_run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
