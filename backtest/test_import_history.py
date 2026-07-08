"""
Fixture test for backtest.import_history — no pytest, plain python.

Builds 3 synthetic session-days of 5m bars (naive UTC timestamps, the
default vendor shape) where day 2 has a known Asia-high fakeout: NY
breaches the Asia high at 09:45 and the breach bar itself closes back
below. Asserts the importer classifies it fakeout/asia_high, tags it
vendor, and — on a second run — treats every day as overlap, changes
nothing, and reports MATCH.

Run: python backtest/test_import_history.py
"""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.import_history import main as import_main  # noqa: E402
from backtest.session_stats import DATASET_VERSION  # noqa: E402

ET = ZoneInfo("America/New_York")

# arbitrary past weekdays guaranteed absent from the real dataset
DAYS = [date(2025, 3, 4), date(2025, 3, 5), date(2025, 3, 6)]  # Tue-Thu

ASIA_HIGH = 20100.0
ASIA_LOW = 20000.0


def bars(day: date, start: time, end: time, lo: float, hi: float,
         day_offset: int = 0) -> list[tuple[datetime, float, float, float, float]]:
    """Flat 5m bars oscillating inside [lo, hi]; first bar prints hi, second lo."""
    out = []
    t = datetime.combine(day + timedelta(days=day_offset), start, tzinfo=ET)
    stop = datetime.combine(day + timedelta(days=day_offset), end, tzinfo=ET)
    i = 0
    while t < stop:
        mid = (lo + hi) / 2
        o = c = mid
        h, l = (hi, mid) if i == 0 else (mid, lo) if i == 1 else (mid + 5, mid - 5)
        out.append((t, o, h, l, c))
        t += timedelta(minutes=5)
        i += 1
    return out


def session_day(day: date, ny_rows: list | None = None) -> list:
    rows = []
    # Asia: prior day 18:00 -> midnight, range [ASIA_LOW, ASIA_HIGH]
    rows += bars(day, time(18, 0), time(23, 55), ASIA_LOW, ASIA_HIGH, day_offset=-1)
    # London: 02:00 -> 05:00, inside Asia (london_sweep = none)
    rows += bars(day, time(2, 0), time(5, 0), 20020.0, 20080.0)
    if ny_rows is None:
        # quiet NY: stays inside every session level
        ny_rows = bars(day, time(9, 30), time(16, 0), 20030.0, 20070.0)
    rows += ny_rows
    return rows


def fakeout_ny(day: date) -> list:
    """NY where bar 3 (09:45) breaches the Asia high and closes back below."""
    rows = []
    t = datetime.combine(day, time(9, 30), tzinfo=ET)
    stop = datetime.combine(day, time(16, 0), tzinfo=ET)
    i = 0
    price = 20050.0
    while t < stop:
        if i == 3:
            o, h, l, c = 20080.0, 20110.0, 20075.0, 20085.0  # breach + reclaim close
        else:
            o = c = price
            h, l = price + 5, price - 5
            if i > 3:
                price = max(20010.0, price - 2.0)  # drift down = short MFE
        rows.append((t, o, h, l, c))
        t += timedelta(minutes=5)
        i += 1
    return rows


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for t, o, h, l, c in rows:
            utc = t.astimezone(timezone.utc).replace(tzinfo=None)
            f.write(f"{utc.isoformat()},{o},{h},{l},{c},100\n")


def main() -> int:
    rows = (session_day(DAYS[0])
            + session_day(DAYS[1], ny_rows=fakeout_ny(DAYS[1]))
            + session_day(DAYS[2]))

    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "vendor.csv"
        dataset = Path(td) / "dataset.json"
        write_csv(csv, rows)
        dataset.write_text(json.dumps(
            {"meta": {"dataset_version": DATASET_VERSION, "instrument": "NQ=F"},
             "sessions": {}}), encoding="utf-8")

        # dry-run first: must plan 3 appends and write nothing
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert import_main([str(csv), "--dry-run", "--dataset", str(dataset)]) == 0
        assert "would append" in buf.getvalue(), buf.getvalue()
        assert json.loads(dataset.read_text())["sessions"] == {}, "dry-run wrote!"

        # real import
        assert import_main([str(csv), "--dataset", str(dataset)]) == 0
        ds = json.loads(dataset.read_text(encoding="utf-8"))
        assert len(ds["sessions"]) == 3, f"expected 3 sessions, got {len(ds['sessions'])}"
        rec = ds["sessions"][DAYS[1].isoformat()]
        assert rec["source"] == "vendor", rec["source"]
        ft = rec["first_touch"]["asia_high"]
        assert ft is not None and ft["kind"] == "fakeout", ft
        assert ft["time"] == "09:45", ft["time"]
        assert rec["asia_high"] == ASIA_HIGH, rec["asia_high"]

        # second run: every day overlaps -> keep records, report MATCH, change 0
        before = dataset.read_text(encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert import_main([str(csv), "--dataset", str(dataset)]) == 0
        out = buf.getvalue()
        assert "MATCH" in out and "MISMATCH" not in out, out
        assert dataset.read_text(encoding="utf-8") == before, "overlap changed records"

    print("test_import_history: all assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
