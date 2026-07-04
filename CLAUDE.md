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
   - Deliver a full market read: gamma regime, nearest levels above/below, session AMD context, intraday bias

4. **Start the 2-minute monitoring loop** using CronCreate:
   - cron: `*/2 * * * 1-5` (every 2 minutes, weekdays only)
   - recurring: true
   - The cron prompt must: capture a chart screenshot + fetch GEX pine labels, compare to last known state, and report ONLY if notable (level break, gamma regime flip, big opening candle, opening range establishing). If nothing notable, say "No significant change — monitoring."
   - Save the CronCreate job ID and tell the user they can stop monitoring with `CronDelete <id>`

5. **Deliver the morning brief summary** from `results/morning_brief.json` — prices, sectors, GEX levels map, AMD session context, news, intraday bias.

**Also trigger step 3+4+5 (skip step 1+2) when** the user says "session ready" or "all logged in" mid-session after having already launched Chrome manually.

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
