"""TiltGuard CLI -- arm a revenge-trade cooldown or check daily P&L stops.

Usage (from the repo root):
    python -m tiltguard.main arm                # cooldown now, config duration
    python -m tiltguard.main arm --minutes 15   # override duration
    python -m tiltguard.main check              # arm only if a daily stop is hit

`check` reads data/tradeify_state.json (maintained by the trading session
protocol via roll_day/check_guardrail) -- no screen reading in v1. Its
verdict is only as fresh as that file; the printed date shows staleness.
"""

import argparse
import json
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
CONFIG_PATH = PKG_DIR / "config.json"
DEFAULT_STATE_PATH = REPO_ROOT / "data" / "tradeify_state.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def arm(cfg: dict, minutes: float | None = None) -> None:
    from tiltguard.overlay import run_overlay

    duration = cfg["cooldown_minutes"] if minutes is None else minutes
    run_overlay(duration, cfg["rules_text"], cfg["unlock_sentence"])


def check(cfg: dict, state_path: Path) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    today_pnl = state["last_seen_balance"] - state["day_start_balance"]
    seen = state.get("last_seen_date", "unknown date")
    loss_stop = cfg["daily_loss_stop"]
    win_stop = cfg["daily_win_stop"]

    if today_pnl <= loss_stop:
        print(f"DAILY LOSS STOP HIT: today {today_pnl:+.2f} (stop {loss_stop:+.2f}, "
              f"state from {seen}) -- arming {cfg['cooldown_minutes']}-minute cooldown.")
        arm(cfg)
    elif today_pnl >= win_stop:
        print(f"DAILY WIN STOP HIT: today {today_pnl:+.2f} (stop {win_stop:+.2f}, "
              f"state from {seen}) -- arming {cfg['cooldown_minutes']}-minute cooldown. "
              f"Bank the day.")
        arm(cfg)
    else:
        print(f"within limits: today {today_pnl:+.2f} "
              f"(loss stop {loss_stop:+.2f}, win stop {win_stop:+.2f}, state from {seen})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tiltguard", description="Revenge-trade cooldown enforcer.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_arm = sub.add_parser("arm", help="start a cooldown overlay now")
    p_arm.add_argument("--minutes", type=float, default=None,
                       help="override cooldown duration from config")

    p_check = sub.add_parser(
        "check", help="check daily P&L against stops; arm a cooldown if breached")
    p_check.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH,
                         help="path to tradeify state file (override for testing)")

    args = parser.parse_args(argv)
    cfg = load_config()
    if args.command == "arm":
        arm(cfg, args.minutes)
    else:
        check(cfg, args.state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
