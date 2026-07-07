# BLUEPRINT 2: SweepStats daily pipeline — compound the dataset + produce the daily post

**BUILDER:** Claude Sonnet, working alone, cold start, cannot ask questions.
(Touches a working data engine and Windows Task Scheduler — Sonnet.)

## GOAL

One command, `python -m backtest.daily_stats_run`, that (1) appends
yesterday's session record to the SweepStats dataset, (2) regenerates the
summary + shareable card, (3) renders the card to a PNG, and (4) writes a
ready-to-paste social caption — plus a scheduled task that runs it every
weekday evening. The dataset compounds daily with zero manual effort; the
user's only manual act is pasting the PNG + caption to X/Discord.

## CONTEXT THE BUILDER NEEDS

- Files to read first: `backtest/session_stats.py` (the engine — read its
  module docstring for the v1 definitions; entry point pattern
  `python -m backtest.session_stats` updates dataset + writes
  `results/session_stats_summary.json`), `backtest/stats_card.py` (renders
  `results/stats_card.html` from the summary), `CLAUDE.md` ("Repo hygiene":
  derived outputs are gitignored; `backtest/session_stats_dataset.json` is
  TRACKED deliberately — commit it when it grows).
- Existing behavior: `python -m backtest.session_stats` is already
  idempotent per date (existing dates kept). yfinance is the data source;
  it needs network. NQ contract conid constant exists for IBKR
  cross-checks but yfinance is the default path.
- Gotchas: PowerShell 5.1 on this machine — scheduled task creation must
  use `schtasks.exe` syntax, not PowerShell-7-only cmdlets. Chrome exists
  at the default install path and the repo already drives Chrome via CDP
  (`data/gex.py`) — but for PNG rendering use HEADLESS chrome, do not touch
  the port-9222 trading instance.

## CONSTRAINTS

- Must stay inside: new file `backtest/daily_stats_run.py`, new file
  `scripts/schedule_sweepstats.bat`, plus a one-line note in `CLAUDE.md`
  repo-hygiene section gitignoring the new PNG/caption outputs.
- Must not change: `backtest/session_stats.py` definitions (DATASET_VERSION
  stays 1), `backtest/stats_card.py` rendering logic (call it, don't edit
  it), the trading-session Chrome on port 9222.
- Stack: Python 3.14 stdlib + the modules the engine already uses
  (pandas/yfinance are installed). No new pip installs.
- Non-negotiables: the run must be safe to execute twice in a day
  (idempotent), must not commit to git by itself, and must never post to
  any social network itself — it PREPARES the post. DECISION: auto-posting
  needs X/Discord API credentials that don't exist yet; out of scope until
  the user creates them.

## STEP-BY-STEP PLAN

1. Create `backtest/daily_stats_run.py` with `main()`:
   a. Run the dataset update in-process: `from backtest import session_stats`
      and call its existing update entry function (read the module to find
      it — it is the function `main`/equivalent used by `__main__`; call
      that, do not shell out).
   b. Regenerate the card: import and call `backtest.stats_card`'s render
      entry the same way.
   c. Render PNG: invoke headless Chrome —
      `chrome --headless=new --disable-gpu --screenshot=<abs results/stats_card.png> --window-size=900,1200 file:///<abs results/stats_card.html>`
      Locate chrome.exe by checking, in order:
      `%ProgramFiles%\Google\Chrome\Application\chrome.exe`,
      `%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe`,
      `%LocalAppData%\Google\Chrome\Application\chrome.exe`. If none exist,
      skip the PNG with a warning line (caption still written).
   d. Write `results/stats_card_caption.txt`: 3 lines built from the
      summary JSON — line 1 `NQ SweepStats — {n} sessions and counting`;
      line 2 the headline stat, e.g. `{pct}% of first NY touches of
      Asia/London levels are fakeouts`; line 3
      `Today's update: {date} · full card below`. Use real keys from
      `results/session_stats_summary.json` (open it and match the actual
      schema; do not invent key names).
   e. Print a one-screen report: dataset size before/after, headline stat,
      paths written, and the reminder `dataset grew — commit
      backtest/session_stats_dataset.json`.
2. Create `scripts/schedule_sweepstats.bat`: registers the task via
   `schtasks /Create /TN "SweepStats Daily" /TR "python -m backtest.daily_stats_run" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:30 /F`
   with the working directory set via a wrapper line (schtasks has no cwd
   option: the /TR command must be
   `cmd /c cd /d C:\Users\lucap\.vscode-shared\claudecodetest && python -m backtest.daily_stats_run`).
   DECISION: 18:30 ET-ish local — after the 16:00 NY close with margin;
   weekly/weekday schedule avoids weekend no-data runs.
3. Add to `.gitignore`: `results/stats_card.png` and
   `results/stats_card_caption.txt` (regenerated outputs, per repo hygiene).
4. Add one bullet to `CLAUDE.md` repo-hygiene list naming those two files
   as gitignored regenerated outputs.
5. Run `python -m backtest.daily_stats_run` once end-to-end and include its
   printed report in your final summary.

## EXACT INPUTS TO USE

- Kickoff prompt: "Open blueprints/sweepstats-daily-pipeline.md and build it
  exactly as written. Read backtest/session_stats.py and
  backtest/stats_card.py first; call their existing entry functions, do not
  reimplement them."

## DEFINITION OF DONE

- [ ] `python -m backtest.daily_stats_run` exits 0, updates the dataset
      idempotently, and writes `results/stats_card.html`,
      `results/stats_card_caption.txt`, and (if Chrome found)
      `results/stats_card.png`.
- [ ] Running it a second time changes nothing except timestamps (same
      dataset size, no duplicate dates).
- [ ] `results/stats_card_caption.txt` contains real numbers from the
      summary JSON, no placeholders.
- [ ] `scripts/schedule_sweepstats.bat` runs without error and
      `schtasks /Query /TN "SweepStats Daily"` shows the task.
- [ ] `git status` shows only: the two new files, .gitignore, CLAUDE.md,
      and (if it grew) the tracked dataset.

## IF SOMETHING IS UNCLEAR

Smallest safe assumption, `ASSUMPTION:` tag, keep going. If the summary
JSON schema differs from the caption fields named here, adapt the caption
to the real schema rather than editing the engine.
