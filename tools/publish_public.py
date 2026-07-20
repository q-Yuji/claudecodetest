"""publish_public.py — build the public site and push it to static hosting.

Phase 3 of the Situation Room roadmap. Since 2026-07-20 this is a thin
pusher over tools/build_site.py, which assembles the COMPLETE deployable
site (landing, gated /room/, free /sample/, login, legal pages, and the
Cloudflare Pages Functions that implement the paywall) into publish/ and
runs the redaction gate on every page it writes — a gate hit aborts the
build before anything reaches the staging clone.

The publish target is a SEPARATE repository configured in
publish_config.json; this repo (journals, trades, account data) is never
the publish target. remote null = dry run: build + gate only, push
skipped. Go-live steps: GO-LIVE.md.

Run:
  python -m tools.publish_public            # build, gate, push if configured
  python -m tools.publish_public --dry-run  # never push, even if configured
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tools import build_site

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT / "publish_config.json"
STAGE_DIR = build_site.STAGE

ET = ZoneInfo("America/New_York")


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


def push_site(cfg: dict) -> str:
    """Commit the built site in the staging clone and push. The clone is
    created on first use; the build has already written into it."""
    remote = cfg["remote"]
    branch = cfg.get("branch") or "main"

    if not (STAGE_DIR / ".git").exists():
        subprocess.run(["git", "init", "-b", branch, str(STAGE_DIR)],
                       check=True, capture_output=True, timeout=60)
        _git("remote", "add", "origin", remote)
        r = subprocess.run(["git", "-C", str(STAGE_DIR), "pull", "origin",
                            branch], capture_output=True, timeout=120)
        if r.returncode:
            pass  # empty remote — first push creates the branch

    if cfg.get("cname"):
        (STAGE_DIR / "CNAME").write_text(str(cfg["cname"]) + "\n",
                                         encoding="utf-8")

    _git("add", "-A")
    if not _git("status", "--porcelain").stdout.strip():
        return "no changes — site already current"
    stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    _git("commit", "-m", f"publish {stamp}")
    _git("push", "-u", "origin", branch)
    return f"pushed to {remote} ({branch})"


def main() -> None:
    dry = "--dry-run" in sys.argv

    print("=== Situation Room — public site build + publish ===")
    build_site.build()  # gate runs inside; SystemExit on any hit

    cfg = load_config()
    if dry or not cfg.get("remote"):
        why = "--dry-run" if dry else "no remote in publish_config.json"
        print(f"  publish: skipped ({why})")
        return
    print(f"  publish: {push_site(cfg)}")


if __name__ == "__main__":
    main()
