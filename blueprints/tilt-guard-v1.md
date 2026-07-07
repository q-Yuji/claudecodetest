# BLUEPRINT 1: TiltGuard v1 — revenge-trade cooldown enforcer (personal use)

**BUILDER:** Claude Sonnet, working alone, cold start, cannot ask questions.
(Multi-file Python + Windows UI edge cases → Sonnet, not Haiku.)

## GOAL

A Windows tray-less desktop utility, `tiltguard/`, that enforces trading
cooldowns at the *input* level: when armed (by hotkey or by rule), it shows a
full-screen, always-on-top, click-through-proof overlay for a countdown
period with the user's own rules text on it, so a revenge trade cannot be
placed impulsively. v1 is personal-use only (no packaging, no market scan
needed — that is explicitly deferred).

Origin story that defines the requirements (2026-07-07 journal): user went
−$900 in 8 minutes on two chased entries after a stop-out, self-diagnosed
revenge trading, then pre-staged a third market order while $100 from an
account-ending floor. The moment to interrupt is the ~6 minutes after a
stop-out. The rules that matter to him: "after a stop-out, only resting
orders at levels — never a market chase" and "stop after +/−$X on the day."

## CONTEXT THE BUILDER NEEDS

- Files to read first: `scripts/log_trade_hotkey.ahk` (existing AutoHotkey
  global-hotkey pattern in this repo — F11 focuses VS Code and types a
  phrase; TiltGuard's hotkey follows the same style), `data/tradeify_state.json`
  (gitignored; shape below), `CLAUDE.md` (repo conventions).
- `data/tradeify_state.json` shape (real example):
  `{"highest_eod_balance": 0.0, "day_start_balance": 0.0, "last_seen_balance": 3244.24, "last_seen_date": "2026-07-07", "daily_profits": {}}`
- The user trades NQ futures in TradingView (Chrome + TradingView desktop).
  v1 does NOT read the screen and does NOT touch the broker — it only
  controls a cooldown overlay. Detection stays manual (hotkey) in v1.
- Gotchas: Python on this machine is 3.14 (`python` on PATH). No package
  manager in the repo — stdlib only unless unavoidable. tkinter IS available
  and is the right tool for the overlay (stdlib, `-topmost`, `-fullscreen`).
  Multiple monitors exist (2): the overlay must cover BOTH (enumerate
  monitors via `ctypes` + `EnumDisplayMonitors`, spawn one Toplevel per
  monitor).

## CONSTRAINTS

- Must stay inside: new folder `tiltguard/` + one new `.ahk` file in
  `scripts/` + one line added to `.gitignore` if a state file is created.
- Must not change: any existing Python module, `my_trades.csv`,
  `data/tradeify_account.py`, existing `.ahk` scripts.
- Stack: Python 3.14 stdlib only (tkinter, ctypes, json, argparse,
  datetime). AutoHotkey v1 syntax for the hotkey script (match the
  existing `log_trade_hotkey.ahk` style).
- Non-negotiables: the overlay must be dismissible ONLY by typing a full
  deliberate sentence (see EXACT INPUTS) — not by a click or Escape. It
  must never block Ctrl+Alt+Del or the ability to kill the process from
  Task Manager (safety: this is a friction device, not a jail).

## STEP-BY-STEP PLAN

1. Create `tiltguard/config.json` — user-editable rules:
   `{"cooldown_minutes": 10, "daily_loss_stop": -900.0, "daily_win_stop": 1500.0, "unlock_sentence": "I am flat and I will only enter with a resting order at a level.", "rules_text": ["No market orders after a stop-out.", "Re-entry = resting order at a level, or nothing.", "The thesis being right does not make the entry right."]}`
   DECISION: config is JSON not YAML (stdlib json, no dependency).
2. Create `tiltguard/overlay.py` — `run_overlay(minutes: float, rules_text: list[str], unlock_sentence: str) -> None`:
   tkinter root withdrawn; one fullscreen `Toplevel` per monitor
   (`ctypes.windll.user32` EnumDisplayMonitors for rects), `-topmost` True,
   re-asserted every 2s via `.after` (TradingView also uses topmost);
   near-black background `#0d0d0d`, countdown `mm:ss` in huge type, rules
   text lines below it, and an Entry widget at the bottom: typing the exact
   `unlock_sentence` (case-insensitive, whitespace-normalized) closes the
   overlay early; otherwise it closes when the countdown hits 0.
   DECISION: unlock-by-sentence instead of hard lock — forces a deliberate
   act without being dangerous.
3. Create `tiltguard/main.py` — CLI:
   - `python -m tiltguard.main arm` → start cooldown now (reads config).
   - `python -m tiltguard.main arm --minutes 15` → override duration.
   - `python -m tiltguard.main check` → read `data/tradeify_state.json`;
     compute `today_pnl = last_seen_balance - day_start_balance`; if
     `today_pnl <= daily_loss_stop` or `>= daily_win_stop`, print the
     verdict and arm a cooldown of `cooldown_minutes`; else print
     "within limits" and exit 0. DECISION: `check` piggybacks on the
     guardrail state file the trading protocol already maintains — no
     screen reading in v1.
   - Package marker `tiltguard/__init__.py` (empty).
4. Create `scripts/tiltguard_hotkey.ahk` — global hotkey `F9` runs
   `python -m tiltguard.main arm` in the repo folder, hidden window.
   Copy the header-comment style of `scripts/log_trade_hotkey.ahk` and note
   in the header: add a Startup-folder shortcut manually, same as F11 script.
   DECISION: F9 (F10/F11/F12 have common conflicts; F11 already taken).
5. Append to `CLAUDE.md` under the trade-logging section a short
   "## TiltGuard" subsection: what F9 does, the `check` command, and that
   Claude should suggest `python -m tiltguard.main check` when a stop-out
   is logged during a session.
6. Test script `tiltguard/smoke_test.py`: launches the overlay for 0.1
   minutes on all monitors, asserts it closes by itself; run it.

## EXACT INPUTS TO USE

- Kickoff prompt for the builder: "Open blueprints/tilt-guard-v1.md and
  build it exactly as written. Work only in tiltguard/, scripts/, and the
  one CLAUDE.md subsection it names."
- Unlock sentence (verbatim, is also the default in config):
  `I am flat and I will only enter with a resting order at a level.`
- Overlay rules text: the three strings in step 1's config JSON, verbatim.

## DEFINITION OF DONE

- [ ] `python -m tiltguard.main arm --minutes 0.1` shows a fullscreen
      topmost overlay on BOTH monitors and closes itself after ~6s.
- [ ] While the overlay is up, typing the unlock sentence into its entry
      box closes it early; typing anything else does not.
- [ ] `python -m tiltguard.main check` with the real state file prints a
      verdict and exits 0 without touching the overlay when within limits.
- [ ] `python tiltguard/smoke_test.py` exits 0.
- [ ] F9 hotkey script exists, parses in AutoHotkey, and its header
      documents Startup installation.
- [ ] No dependency was added; no existing file except CLAUDE.md changed.

## IF SOMETHING IS UNCLEAR

Smallest safe assumption, `ASSUMPTION:` tag at top of your summary, keep
going. Do not add broker/screen integration, sounds, or tray icons — v2 ideas
live in the journal, not this build.
