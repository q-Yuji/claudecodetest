"""publish_public.py — push the PUBLIC Situation Room edition to static hosting.

Phase 3 of the Situation Room roadmap: after each pipeline run, the
redacted product page gets a live URL instead of living as a screenshot.

Safety model (the load-bearing part):
  1. Only ever publishes results/situation_room_public.html — the edition
     rendered with --public (price-derived levels, no GEX, no account
     panels). It is re-rendered here, never trusted from disk.
  2. A REDACTION GATE scans the rendered HTML for forbidden markers
     (GEX/gamma vocabulary, prop-firm and broker identifiers, ledger and
     account-state language). Any hit aborts the publish with the matched
     term. False positives fail safe — loosen the list consciously, in
     a reviewed commit, never at runtime.
  3. The publish target is a SEPARATE git repository configured in
     publish_config.json. This repo (journals, trades, account data) is
     never the publish target. No config -> dry run: render + gate only.

Config (publish_config.json at repo root, tracked):
  {
    "remote": "https://github.com/<org>/<page-repo>.git",  // null = dry run
    "branch": "main",
    "cname": null            // custom domain, written as CNAME if set
  }

Run:
  python -m tools.publish_public            # render, gate, push if configured
  python -m tools.publish_public --dry-run  # never push, even if configured
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
PUBLIC_HTML = RESULTS / "situation_room_public.html"
CONFIG_FILE = ROOT / "publish_config.json"
STAGE_DIR = ROOT / "publish"  # gitignored working clone of the page repo

ET = ZoneInfo("America/New_York")

# Markers that must NEVER appear on the public page. Vocabulary, not just
# secrets: if the GEX ladder or an account panel ever leaks into the public
# edition, some of these words come with it.
FORBIDDEN = [
    "gex", "gamma", "call wall", "put wall",          # licensed-feed data
    "mffu", "tradeify", "mffuevbldr", "u24694898",    # firms + account ids
    "my funded futures", "apex",                      # firms
    "ledger", "payout", "cost_usd", "prop_ledger",    # account economics
    "trailing floor", "buffer", "roll_day",           # guardrail language
    "tradovate", "ibkr", "interactive brokers",       # brokers
    "tradeify_state", "drawdown",                     # state + risk language
]


def render_public() -> str:
    """Re-render the public edition and return its HTML. Never publishes
    whatever happened to be on disk."""
    subprocess.run([sys.executable, str(ROOT / "situation_room.py"), "--public"],
                   check=True, capture_output=True, timeout=300, cwd=ROOT)
    return PUBLIC_HTML.read_text(encoding="utf-8")


def redaction_gate(html: str) -> list[str]:
    """Return the forbidden markers present in the page (empty = clean).
    Case-insensitive; checked against the raw HTML so attributes and
    comments are covered too."""
    low = html.lower()
    return [term for term in FORBIDDEN if term in low]


def load_config() -> dict:
    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(STAGE_DIR), *args],
                          check=True, capture_output=True, text=True,
                          timeout=120)


def push_page(html: str, cfg: dict) -> str:
    """Commit the page into the staging clone of the page repo and push.
    The staging dir is created/cloned on first use."""
    remote = cfg["remote"]
    branch = cfg.get("branch") or "main"

    if not (STAGE_DIR / ".git").exists():
        STAGE_DIR.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--branch", branch, "--single-branch",
                        remote, str(STAGE_DIR)],
                       check=False, capture_output=True, timeout=120)
        if not (STAGE_DIR / ".git").exists():  # empty repo — start fresh
            subprocess.run(["git", "init", "-b", branch, str(STAGE_DIR)],
                           check=True, capture_output=True, timeout=60)
            _git("remote", "add", "origin", remote)
    else:
        _git("fetch", "origin", branch)
        _git("reset", "--hard", f"origin/{branch}")

    (STAGE_DIR / "index.html").write_text(html, encoding="utf-8")
    (STAGE_DIR / ".nojekyll").write_text("", encoding="utf-8")
    if cfg.get("cname"):
        (STAGE_DIR / "CNAME").write_text(str(cfg["cname"]) + "\n",
                                         encoding="utf-8")

    _git("add", "-A")
    if not _git("status", "--porcelain").stdout.strip():
        return "no changes — page already current"
    stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    _git("commit", "-m", f"publish {stamp}")
    _git("push", "-u", "origin", branch)
    return f"pushed to {remote} ({branch})"


def main() -> None:
    dry = "--dry-run" in sys.argv

    print("=== Situation Room — public publish ===")
    html = render_public()
    print(f"  rendered: {PUBLIC_HTML.name} ({len(html):,} bytes)")

    hits = redaction_gate(html)
    if hits:
        print(f"  REDACTION GATE FAILED — forbidden markers on the public "
              f"page: {hits}")
        print("  NOT publishing. Fix the leak (or consciously amend "
              "FORBIDDEN in a reviewed commit).")
        raise SystemExit(1)
    print(f"  redaction gate: clean ({len(FORBIDDEN)} markers checked)")

    cfg = load_config()
    if dry or not cfg.get("remote"):
        why = "--dry-run" if dry else "no remote in publish_config.json"
        print(f"  publish: skipped ({why})")
        return
    print(f"  publish: {push_page(html, cfg)}")


if __name__ == "__main__":
    main()
