"""
daily_stats_run.py — One-shot SweepStats daily pipeline.

Appends any new sessions to the dataset (via backtest.session_stats),
regenerates the summary + shareable card (via backtest.stats_card), renders
the card to results/stats_card.png with headless Chrome, and writes a
ready-to-paste caption to results/stats_card_caption.txt. Idempotent: safe
to run twice in a day. Never commits, never posts anywhere — it PREPARES
the daily post; pasting the PNG + caption to X/Discord stays manual.

Run:
  python -m backtest.daily_stats_run

Scheduled every weekday 18:30 by scripts/schedule_sweepstats.bat
(schtasks task "SweepStats Daily").
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date
from pathlib import Path

from backtest import session_stats, stats_card

RESULTS_DIR = Path(__file__).parent.parent / "results"
SUMMARY_FILE = RESULTS_DIR / "session_stats_summary.json"
CARD_HTML = RESULTS_DIR / "stats_card.html"
CARD_PNG = RESULTS_DIR / "stats_card.png"
CAPTION_FILE = RESULTS_DIR / "stats_card_caption.txt"

CHROME_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    / r"Google\Chrome\Application\chrome.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / r"Google\Chrome\Application\chrome.exe",
    Path(os.environ.get("LocalAppData", "")) / r"Google\Chrome\Application\chrome.exe",
]


def _find_chrome() -> Path | None:
    for p in CHROME_CANDIDATES:
        if p.is_file():
            return p
    return None


def _render_png() -> bool:
    """Render the card HTML to PNG with a fresh headless Chrome.

    Never touches the port-9222 trading Chrome — this is a separate,
    short-lived headless process.
    """
    chrome = _find_chrome()
    if chrome is None:
        print("  WARNING: chrome.exe not found — skipping PNG render")
        return False
    subprocess.run(
        [str(chrome), "--headless=new", "--disable-gpu",
         f"--screenshot={CARD_PNG.resolve()}", "--window-size=900,1200",
         CARD_HTML.resolve().as_uri()],
        check=True, capture_output=True, timeout=120)
    return True


def _write_caption(summary: dict) -> str:
    ft = summary["first_touch"]
    tot_touch = sum(v["touched"] for v in ft.values())
    tot_fake = sum(v["fakeout"] for v in ft.values())
    pct = round(100.0 * tot_fake / tot_touch) if tot_touch else 0
    n = summary["sample"]["sessions"]
    caption = (
        f"NQ SweepStats — {n} sessions and counting\n"
        f"{pct}% of first NY touches of Asia/London levels are fakeouts\n"
        f"Today's update: {date.today().isoformat()} · full card below\n"
    )
    CAPTION_FILE.write_text(caption, encoding="utf-8")
    return caption


def main() -> None:
    size_before = len(session_stats.load_dataset()["sessions"])

    session_stats.main()   # update dataset + write summary (idempotent)
    stats_card.main()      # summary -> results/stats_card.html

    size_after = len(session_stats.load_dataset()["sessions"])
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    caption = _write_caption(summary)
    png_ok = _render_png()

    print("=== SweepStats daily run ===")
    print(f"  dataset: {size_before} -> {size_after} sessions "
          f"({size_after - size_before} new)")
    print(f"  headline: {caption.splitlines()[1]}")
    print(f"  card    : {CARD_HTML}")
    if png_ok:
        print(f"  png     : {CARD_PNG}")
    print(f"  caption : {CAPTION_FILE}")
    if size_after > size_before:
        print("  dataset grew — commit backtest/session_stats_dataset.json")


if __name__ == "__main__":
    main()
