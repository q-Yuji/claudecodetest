"""
friction_review.py — Rank a week of friction-journal entries into a shortlist.

Parses journal/frictions.md (entries appended by the F8 hotkey,
scripts/friction_hotkey.ahk), groups related entries — same app title, or
descriptions sharing 3+ common words of 4+ chars — and prints a ranked
table: count, app, representative description, dates seen.

Run:
  python tools/friction_review.py [--since YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

FRICTIONS_FILE = Path(__file__).parent.parent / "journal" / "frictions.md"

# only lines at column 0 count — the indented format example in the header
# deliberately doesn't match
ENTRY_RE = re.compile(
    r"^- (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) \| \[(.*?)\] \| (.+)$")

FILTER_REMINDER = ("Filter: my daily friction AND a reachable crowd "
                   "shares it → demand / wedge / distribution.")


def parse_entries(text: str, since: str | None) -> list[dict]:
    entries = []
    for line in text.splitlines():
        m = ENTRY_RE.match(line)
        if not m:
            continue
        d, t, app, desc = m.groups()
        if since and d < since:
            continue
        entries.append({"date": d, "time": t, "app": app.strip(),
                        "desc": desc.strip()})
    return entries


def _keywords(desc: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", desc.lower()) if len(w) >= 4}


def group_entries(entries: list[dict]) -> list[list[dict]]:
    """Union-find: merge on same app title (case-insensitive) or on
    descriptions sharing 3+ common words of 4+ chars."""
    parent = list(range(len(entries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    kw = [_keywords(e["desc"]) for e in entries]
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i]["app"].lower() == entries[j]["app"].lower():
                union(i, j)
            elif len(kw[i] & kw[j]) >= 3:
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, e in enumerate(entries):
        groups.setdefault(find(i), []).append(e)
    return sorted(groups.values(), key=len, reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="friction_review")
    parser.add_argument("--since", metavar="YYYY-MM-DD", default=None,
                        help="only entries on/after this date")
    parser.add_argument("--file", type=Path, default=FRICTIONS_FILE,
                        help="frictions file override (testing)")
    args = parser.parse_args(argv)

    # the reminder line contains "→"; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    entries = parse_entries(args.file.read_text(encoding="utf-8"), args.since)
    if not entries:
        print("no entries yet — press F8 when something feels like friction")
        print(FILTER_REMINDER)
        return 0

    print(f"{len(entries)} entries, ranked by recurrence:\n")
    print(f"{'#':>3}  {'app':<32} {'dates':<24} representative")
    for grp in group_entries(entries):
        app = Counter(e["app"] for e in grp).most_common(1)[0][0]
        dates = sorted({e["date"] for e in grp})
        date_str = ", ".join(d[5:] for d in dates)
        rep = max((e["desc"] for e in grp), key=len)
        print(f"{len(grp):>3}  {app[:32]:<32} {date_str[:24]:<24} {rep}")
    print()
    print(FILTER_REMINDER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
