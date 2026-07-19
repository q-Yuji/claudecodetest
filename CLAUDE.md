# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Jarvis is a static front-end dashboard. No build step, no dependencies, no package manager. Open `index.html` directly in a browser to run it.

```bash
open index.html
```

## Architecture

Three files, each with a distinct responsibility:

- **`index.html`** — all markup and layout. Stat cards and activity chart bars are hardcoded here; task list items are also seeded here but can be added dynamically via JS.
- **`style.css`** — all visual design. Theming is done entirely through CSS custom properties defined in `:root` (colors, sidebar width). The activity bar heights are driven by the `--h` CSS variable set inline on each `.bar` element.
- **`app.js`** — all interactivity: live clock/greeting (updates every second via `setInterval`), sidebar nav active-state toggling, and task list management (add task, toggle done state).

## Git workflow

Commit work to git regularly throughout a session — after each meaningful change, not just at the end. Push to GitHub (`git push`) so there is always an up-to-date remote backup. Use clean, descriptive commit messages that explain *why* the change was made. A Stop hook in `~/.claude/settings.json` auto-pushes on session end, but don't rely on that alone — commit and push at logical checkpoints.

## Repo hygiene

- `__pycache__/` and `*.pyc` are gitignored — never commit compiled bytecode.
- `results/morning_brief.html`, `results/morning_brief.json`, `results/amd_review.html`, `results/amd_review.json`, `results/gex_spy.png`, and `results/gex_levels.json` are gitignored: they're fully overwritten every run and have no diff value. The durable record of trading activity lives in `journal/` and `my_trades.csv`, not these.
- `results/chinaV3_indicator.pine` and `results/chinaV3_results.json` **are** tracked deliberately — the Pine script is source code, and the backtest results are worth versioning alongside strategy changes so performance can be diffed across iterations.
- `backtest/session_stats_dataset.json` **is** tracked deliberately — it's the append-only per-session dataset behind the SweepStats product idea (`backtest/session_stats.py`); the whole point is that it compounds across runs. Its derived outputs (`results/session_stats_summary.json`, `results/stats_card.html`, `results/stats_card.png`, `results/stats_card_caption.txt`, and the evening scoreboard set `results/scoreboard.json`, `results/scoreboard_card.html/.png`, `results/scoreboard_caption.txt`) are gitignored like other regenerated files.
- When adding new generated output files, default to gitignoring them unless there's a specific reason to version the history.

## Trading session startup — AUTOMATED PROTOCOL

**Trigger phrases** — when the user says any of the following, execute the full startup sequence below automatically without asking for confirmation:
- "start session" / "new trading day" / "start trading" / "launch trading"
- "its a new trading day" / "session ready" / "ready to trade" / "market opens soon"
- Any greeting that implies a new trading session is beginning

**Full startup sequence (run every step, in order):**

1. **Run the launch script:**
   ```
   python start_session.py
   ```
   This opens Chrome with TradingView + GEX Suite + IBKR Gateway and generates the morning brief.

2. **Wait for user to confirm they've logged in** to IBKR, TradingView, and GEX Suite. If they say "done", "logged in", "all logged in", or similar — proceed to step 3.

3. **Pull live chart state** using TradingView MCP:
   - `mcp__tradingview__tv_health_check` — confirm CDP connected
   - `mcp__tradingview__capture_screenshot` (region: chart) — grab current chart
   - `mcp__tradingview__data_get_pine_labels` (study_filter: GexSuite) — get live GEX levels
   - Write the captured labels to `results/gex_levels.json` (same schema as existing file), then run `python -m backtest.append_gex_history` to bank the day into `backtest/gex_history.json` — do this every session morning; it's the only chance to capture that day's NQ gamma levels before they age out.
   - Deliver a full market read: gamma regime, nearest levels above/below, session AMD context, intraday bias
   - Also screenshot the Tradovate panel to read the current account balance, then call `roll_day(balance)` from `data/tradeify_account.py` to establish today's starting balance for the daily-drawdown floor (see "Tradeify account guardrail" below). This only works if run close to session start — a late first call will use a moved balance as the day-start baseline.

4. **Start the 2-minute monitoring loop** using CronCreate:
   - cron: `*/2 * * * 1-5` (every 2 minutes, weekdays only)
   - recurring: true
   - The cron prompt must: capture a chart screenshot + fetch GEX pine labels, compare to last known state, and report ONLY if notable (level break, gamma regime flip, big opening candle, opening range establishing). If nothing notable, say "No significant change — monitoring."
   - Save the CronCreate job ID and tell the user they can stop monitoring with `CronDelete <id>`

5. **Deliver the morning brief summary** from `results/morning_brief.json` — prices, sectors, GEX levels map, AMD session context, news, intraday bias.

**Also trigger step 3+4+5 (skip step 1+2) when** the user says "session ready" or "all logged in" mid-session after having already launched Chrome manually.

## On-demand trade logging

**Trigger phrases** — "log that trade" / "log this trade" / "record that trade" or similar, said right after executing (or closing) an NQ futures trade on TradingView via the Tradovate broker panel (Tradeify prop account).

**Global hotkey:** `F11` runs `scripts/log_trade_hotkey.ahk` (AutoHotkey, auto-starts on login via a Startup-folder shortcut) — it activates the VS Code window running Claude Code, focuses its active terminal, and types/submits "log that trade" for you. Works from any application (e.g. while TradingView has focus). Note: F11 is normally the fullscreen-toggle key in Chrome/TradingView/VS Code — it stops doing that while this script is running. Also, it focuses whichever terminal tab is currently active in VS Code, so keep the Claude Code tab as the last one you used if you have other terminal tabs open.

**Why on-demand, not a background poll:** Tradovate's API isn't available for Tradeify's sim-funded accounts, so the only way to read a fill is visually — screenshotting the Tradovate panel in TradingView and reading it, the same way GEX pine labels are read. Polling that on a timer (like the 2-minute GEX monitor) would burn a screenshot + vision read on every cycle regardless of whether a trade happened — expensive across a full session. Doing it only when the user asks means the cost scales with actual trades taken, not with elapsed time.

**Steps when triggered:**
1. `mcp__tradingview__capture_screenshot` — capture the Tradovate trading panel (Positions/Orders).
2. Read off: direction, entry or exit price, contracts, approximate time.
3. Append a draft row to `my_trades.csv` with those fields filled in and the qualitative columns (`setup_notes`, `gex_level_type`, `level_strength`, `smt_confluence`, `gamma_regime`) left blank.
4. Tell the user the row was added and ask them to fill in the qualitative context (what they saw, which GEX level, SMT confluence, etc.) — either now or after the session.
5. Flag that the entry/exit values were read visually off a screenshot, not pulled from a data feed — worth a quick spot-check against the actual fill before treating the row as ground truth for analytics.

## TiltGuard (revenge-trade cooldown)

`F9` (global hotkey, `scripts/tiltguard_hotkey.ahk`) arms a fullscreen topmost cooldown overlay on both monitors — countdown + the user's rules text — dismissible only by the countdown ending or by typing the unlock sentence from `tiltguard/config.json`. CLI equivalents: `python -m tiltguard.main arm [--minutes N]`, and `python -m tiltguard.main check` which reads `data/tradeify_state.json` and arms a cooldown automatically if today's P&L has hit the configured daily loss/win stop. When a stop-out is logged during a session (via "log that trade"), suggest running `python -m tiltguard.main check`.

## Market events layer

`data/market_events.py` classifies flow/event days for any date — rule-computed OPEX (third Friday), quad witching, VIX expiration (Wed 30 days before next month's third Friday), plus 2026 CPI/FOMC dates pinned from official schedules (**extend the pinned lists each December**; NFP deliberately absent — its first-Friday rule has exceptions and a wrong flag in the dataset is worse than none). Why it exists (user, 2026-07-19): flow days override structure — the 07-17 intraday V-recovery was OPEX flows, not market structure; CPI usually prints one outsized candle. Consumers: the morning brief calendar (appends OPEX/VIX-exp/quad rows the econ feed doesn't carry), the Situation Room's Today's Script (amber event chips + "pattern odds take a back seat" note) and Record panel (event chip on graded calls), the evening scoreboard card, and `backtest/session_stats.py`, which stamps `events` on every dataset record (auto-backfilled) and reports an `event_days` behavior breakdown in the summary. CLI check: `python -m data.market_events [YYYY-MM-DD]`.

## Conditioning layers (regime + cross-market)

Two walk-forward-safe tags are stamped on the dataset by `backtest/session_stats.py` each run (auto-backfilled where data allows):
- **`regime`** per session — `backtest/regime.py` classifies trend/chop from a 20-session Kaufman efficiency ratio on daily closes, using only closes *before* the session date. Thresholds: ER ≥ 0.35 trend (by sign), ≤ 0.20 chop, else mixed. Validated against the user's account: Apr–May 2026 = trend_up, since Jun 8 = chop. This is the systematized version of the user's regime gate (their STDV strategy works only in chop). CLI: `python -m backtest.regime`.
- **`es_confirmed`** per first-touch episode — was ES already through its own corresponding level when NQ first breached (±1 bar)? False = NQ swept alone ("cross-market divergence" — neutral-quant naming per user veto, never "SMT"). Headline finding (2026-07-19, n=137 touches): diverged sweeps fake out **76% vs 57%** confirmed, with shallower overshoot (~24 vs ~43 pts) and equal bounce — the product's flagship original stat. Episodes older than the ES 5m window stay untagged, never guessed. **The vendor-data purchase should include ES history alongside NQ** so this stat gets the n≈250 treatment.

The summary exposes `cross_market`, `regime`, and `event_days` sections plus `latest_regime`; the Situation Room shows a REGIME chip in Today's Script and a CONDITIONS tab in The Numbers (both editions — all price-derived).

- **`weekend_gap`** per Monday session (added 2026-07-20, first item of the user's own confluence test queue — neutral-quant name, never "NWOG"): gap = Sunday 18:00 ET open − Friday's last 5m close at/before 17:00 ET; fill target = the Friday close; overnight fill vs Monday-NY first-touch classification (fill-and-reject / fill-and-continue via the standard `_first_touch` machinery). Kept OUT of the `first_touch` pools so historical published stats never shift and the ES tagger can't mislabel it. Holiday-Friday weekends are skipped, never guessed. First finding (n=7): 86% of gaps fill during the overnight before NY opens — the "gap as intraday level" framing barely applies to NQ. Surfaced as a WEEKEND GAP block in The Numbers, both editions.

## Situation Room

`python situation_room.py` renders `results/situation_room.html` — the dark war-room "NQ Situation Room" page (SweepStats product wrapper, blueprint #6 + roadmap's Today's Script panel) from `results/morning_brief.json`, `results/gex_levels.json`, and `results/session_stats_summary.json`. Flags: `--png` (renders `results/situation_room.png` via a fresh headless Chrome, never the port-9222 one), `--open` (open in browser), `--public` (product edition → `results/situation_room_public.html/png`: no GEX data, session-liquidity ladder with SweepStats odds instead — the sellable base tier). Read-only over its inputs; regenerate whenever asked ("open the situation room" / "update the war room"). Panels degrade rather than hide: a missing input shows "NO FEED", a >24h-old input dims with an amber "STALE" chip. Both outputs are gitignored.

Both editions include **The Record** panel (roadmap feature 5, USP #1 "the morning call that grades itself"): `backtest/scoreboard.py` re-grades every session's would-be Script call walk-forward from the dataset — each call computed only from sessions before its date, so every grade is out-of-sample and the whole record is reproducible from the dataset's git history (no separate card archive needed). `python -m backtest.scoreboard` prints the record and writes `results/scoreboard.json` (gitignored); `situation_room.py` imports it directly, so the panel is always as fresh as the dataset. The 18:30 pipeline also renders the shareable evening card (`backtest/scoreboard_card.py` → `results/scoreboard_card.png` + `results/scoreboard_caption.txt`) — the second daily content unit alongside the morning stats card; posting stays manual.

The **personal edition only** also renders **The Ledger** (section 05): prop-account economics from `data/prop_ledger.json` (tracked — durable record, hand-curated) — what each eval/funded account cost vs payouts received, pass rate, blown-account lifespans. Fees are recorded facts, never estimates: a missing `cost_usd` renders as unknown and the net is flagged unknowable until filled in. When a new account is bought, blown, passed, or pays out, update the ledger JSON in the same conversation. This panel (and the file's contents) must NEVER appear in the `--public` edition — it is account P&L, which the product page excludes by design. Standing finding at n=53 (2026-07-19): the directional London-pattern call shows **no edge** (41% hit, and it does NOT improve with bucket size or claimed edge), while the fakeout headline **holds** out-of-sample (claimed ~60%, ran 64% over 128 touches). The panel publishes both verdicts — negative results are the anti-guru positioning, not a bug.

## End-of-day journal — AUTOMATED PROTOCOL

**Trigger phrases** — "write my journal" / "journal the day" / "wrap up the session" / "end of day" or similar, near the end of a trading session.

**Steps when triggered:**
1. Run `python make_journal.py` — generates a draft `journal/YYYY-MM-DD.md` from that day's `my_trades.csv` rows (points, gross P&L, R-multiple computed), the Tradeify guardrail state, and same-day market context from `results/morning_brief.json` + `results/gex_levels.json`. It refuses to overwrite an existing entry; pass `--force` only if the user confirms regeneration.
2. Fill in the qualitative placeholder sections (`Execution`, `Exit rationale`, `What went right / what to review`) from what was discussed during the session — trade-logging conversations, monitoring observations. Ask the user about anything not covered rather than inventing narrative.
3. If any `my_trades.csv` rows for today still have blank qualitative columns (`setup_notes`, `gex_level_type`, etc.), prompt the user to fill them now while memory is fresh, and mirror the answers into both the CSV and the journal.
4. Commit the journal entry (journal files are tracked — they're the durable record).

## Prop account guardrail (MFFU Builder)

Current account: **MFFU (My Funded Futures) "Builder" $50k eval**, acct MFFUEVBLDR401004033, bought 2026-07-16 — one of TWO Builder accounts (the second is untracked so far). All prior Tradeify accounts are blown (funded + 4 evals, last one 2026-07-17); historical Tradeify rules live in git history. Balances tracked relative to the $50k base (base = $0). The guardrail module is still `data/tradeify_account.py` (name kept for import stability).

**Eval pass requirements** (phase = "eval"):
- **Profit target:** +$3,000 total — **no consistency rule**; can be passed in a single day. (Balance $53,272.24 was hit 2026-07-17 — pass pending MFFU's EOD sync.)
- **No payouts during eval.**

**Drawdown rules:**
- **Trailing EOD drawdown:** $2,000 (4%) below the highest EOD closing balance — confirmed off the MFFU stats dashboard. No daily/intraday loss limit.

**Funded-phase rules (per user, 2026-07-17 — details to confirm once funded):**
- 50% consistency rule; max payout $2,000 per request, requiring roughly a +$4,000 balance.

**MFFU dashboard:** https://myfundedfutures.com/stats (open it in the debug Chrome; readable via CDP like the Tradovate web platform — see `scratchpad` capture scripts pattern: activate tab via `/json/activate/<id>`, then `Page.captureScreenshot`). The Tradovate web platform (trader.tradovate.com) in the debug Chrome is the fills ground truth when the TradingView broker panel is down or lying.

**Trigger phrases** — "check my buffer" / "check my drawdown" / "how far from passing" or similar.

**Steps when triggered:**
1. Screenshot the Tradovate panel and read the current balance.
2. Call `check_guardrail(balance)` from `data/tradeify_account.py` — it also updates the persisted state (`data/tradeify_state.json`, gitignored).
3. Report: distance to trailing floor, distance to daily floor, and eval pass progress (profit still needed, today's consistency-clean pass window).
4. If either floor is close (use judgment — e.g. within ~20% of the drawdown limit) or already breached, say so plainly rather than burying it in the numbers.

**Accuracy depends on `roll_day` having been called at session start** (see step 3 of the trading session startup protocol) — the daily floor is only correct if today's starting balance was captured before any trades that day.

## Trading assistant context

This repo also contains a trading assistant for NQ futures + options. Key files:
- `morning_brief.py` — run before market open, generates `results/morning_brief.html` dashboard + `results/morning_brief.json`
- `data/ibkr.py` — IBKR Client Portal Gateway REST client (authenticate at https://localhost:5000)
- `data/gex.py` — captures GEX Suite screenshot via CDP from Chrome on port 9222
- `scripts/launch_trading_chrome.bat` — starts IBKR gateway + Chrome together (run this first each session)

### User trading profile
- Trades NQ futures and options on IBKR (account U24694898)
- Has 2 monitors: TradingView on one, GEX Suite on the other
- Wants pre-session NQ reads that synthesize GEX levels + TradingView chart + news + IBKR positions

### GEX Suite indicator — level reference

**Level types ranked by importance:**
1. **Gamma Flip** — above = positive gamma (choppy, mean-reverting, fade/scalp); below = negative gamma (trending, momentum, breakdowns accelerate)
2. **0DTE Call Wall / Put Wall** — same-day expiration, highly reactive intraday magnets/barriers
3. **Call Wall** — highest call-side gamma concentration; mechanical selling pressure; break through flips it to support
4. **Put Wall** — highest put-side gamma concentration; structural floor; fails in bearish conditions → rapid downside
5. **Session Ceiling / Floor** — expected daily range from implied vol + positioning; break outside = explosive move
6. **Gamma Levels 1–10** — ranked reaction zones; Level 1 strongest; price steps between them
7. **Correlated Levels 1–10** — cross-asset influence projected onto chart; most powerful at confluence with Gamma levels

**Confluence rule:** When multiple level types stack at the same price, probability of reaction is highest. Anchor analysis to confluence zones first.

**+ strength modifier** (not in official docs — user confirmed):
- Levels can show `+`, `++`, or `+++` on the chart label (not every day, only some levels)
- `+++` = maximum strength / highest conviction
- Prioritise `+++` levels first regardless of level type — a `+++ Correlated Level` outweighs a plain `Call Wall`

**Supported tickers:** ES1!, NQ1!, GC1!, SI, QQQ, SPY, GLD, SIL, SPX, NDX, VIX, NVDA, AAPL, AMZN, GOOG/GOOGL, META, MSFT, TSLA

## Design conventions

- Dark theme only. Color palette lives in `:root` in `style.css` — always use those variables, never hardcode colors.
- Layout is sidebar (fixed, 200px) + `.main` (flexbox column, `margin-left: var(--sidebar-w)`).
- New dashboard sections/panels should use the `.card` class.
- The `.hidden` utility class (`display: none`) is used to show/hide the add-task form.
- Stat change indicators use `.positive`, `.negative`, `.neutral` modifier classes on `.stat-change`.
