# BLUEPRINT 4: GEX history capture — stop letting live levels age out

**BUILDER:** Claude Sonnet, working alone, cold start, cannot ask questions.
(Small, but edits the session protocol in CLAUDE.md and must understand the
MCP boundary — Sonnet.)

## GOAL

Every trading morning's live GexSuite levels get appended to the tracked
`backtest/gex_history.json` automatically as part of the session-startup
protocol, via a small merge script — so NQ gamma levels stop aging out of
the indicator's ~2-week replay lookback and the level-reaction dataset can
grow past its current gap-riddled state. (This was proposed 2026-07-05 and
approved 2026-07-07: DECISION — fold it into startup step 3.)

## CONTEXT THE BUILDER NEEDS

- Files to read first: `backtest/gex_history.json` (existing per-date
  schema — match it exactly), `CLAUDE.md` section "Trading session startup
  — AUTOMATED PROTOCOL" step 3, `results/gex_levels.json` (the freshest
  live-capture shape: `{"timestamp", "symbol", "current_price", "levels":
  [{"name", "price"}], "session_amd": {...}}`).
- THE MCP BOUNDARY (critical): the live labels come from the TradingView
  MCP tool `mcp__tradingview__data_get_pine_labels` (study_filter
  "GexSuite"), which only Claude can call during a session — a Python
  script cannot. So the design is: Claude captures labels (already does in
  step 3) and writes/refreshes `results/gex_levels.json`; the SCRIPT's job
  is only to merge that file into the history with dedupe. Do not attempt
  to call TradingView from Python.
- Label text carries strength markers like `Γ-2 [++]` or
  `QQQ 720.5 [+++]` — the merge must preserve the full label text; strength
  parsing stays downstream in `amd_session_review.py`.
- Gotcha: `results/gex_levels.json` can be STALE (it held June 24 levels on
  July 7). The merge script must refuse to append when the file's
  timestamp date ≠ today unless `--force-date` is passed.

## CONSTRAINTS

- Must stay inside: new file `backtest/append_gex_history.py`, an edit to
  ONE step of CLAUDE.md's startup protocol, and appends to
  `backtest/gex_history.json`.
- Must not change: `amd_session_review.py`, the schema of existing
  gex_history entries, `data/gex.py`.
- Stack: Python stdlib only.
- Non-negotiables: idempotent (re-running on the same day replaces that
  day's entry rather than duplicating); never deletes prior dates.

## STEP-BY-STEP PLAN

1. Create `backtest/append_gex_history.py` CLI:
   `python -m backtest.append_gex_history [--force-date]`
   a. Load `results/gex_levels.json`; if its timestamp's date ≠ today and
      no `--force-date`, exit 1 with message "gex_levels.json is stale
      (<date>) — recapture step 3 first".
   b. Transform to the gex_history per-date schema (open
      `backtest/gex_history.json` and match its existing entry shape
      exactly — keys, nesting, level record fields).
   c. Insert/replace today's entry, keep the file sorted by date, write
      back with matching indentation.
   d. Print: date written, level count, total dates in history.
2. Edit `CLAUDE.md` startup protocol step 3: after the pine-labels
   capture bullet, add one bullet: "Write the captured labels to
   `results/gex_levels.json` (same schema as existing file), then run
   `python -m backtest.append_gex_history` to bank the day into
   `backtest/gex_history.json` — do this every session morning; it's the
   only chance to capture that day's NQ gamma levels before they age out."
3. Test: run the script with the current (stale) `results/gex_levels.json`
   and confirm it refuses; run with `--force-date` against a COPY of
   gex_history in the scratchpad to confirm merge shape, then run the real
   one only if `results/gex_levels.json` is fresh today (2026-07-07 data
   IS fresh in the session_amd sense but the levels block may be June 24 —
   trust the timestamp field, per the gotcha).

## EXACT INPUTS TO USE

- Kickoff prompt: "Open blueprints/gex-history-capture.md and build it
  exactly as written. Match the existing gex_history.json entry schema —
  open it first."

## DEFINITION OF DONE

- [ ] Script refuses stale input without `--force-date` (exit 1, clear message).
- [ ] Running twice for the same date leaves exactly one entry for that date.
- [ ] Existing historical entries byte-identical after an append (verify
      with git diff showing only the new/replaced date).
- [ ] CLAUDE.md step 3 contains the new bullet, and nothing else in the
      protocol changed.

## IF SOMETHING IS UNCLEAR

Smallest safe assumption, `ASSUMPTION:` tag, keep going. If gex_history.json
has multiple entry shapes across dates, match the NEWEST entry's shape.
