# Blueprints vault

Build-ready specs written by Fable 5 on 2026-07-07 (planning session) so a
cheaper model (Sonnet/Haiku) can build each one cold — open one blueprint,
follow it literally, tick the DEFINITION OF DONE. Format: `_TEMPLATE.md`.

To build one, start a session and say:
> Open blueprints/<name>.md and build it exactly as written.

| # | Blueprint | Builder | Priority / why |
|---|---|---|---|
| 1 | `tilt-guard-v1.md` | Sonnet | Personal safety tool; requirements written by the 2026-07-07 revenge-trading episode. Build first — it protects the funded account. |
| 2 | `sweepstats-daily-pipeline.md` | Sonnet | Makes the SweepStats dataset compound daily with a ready-to-post card — the content-led distribution loop. |
| 3 | `sweepstats-deep-history.md` | Sonnet | 46 sessions → ~250 via a ~$10–30 purchased file; importer is buildable before purchase. |
| 4 | `gex-history-capture.md` | Sonnet | Small; stops daily NQ gamma levels aging out of the ~2-week indicator lookback. Approved 2026-07-07. |
| 5 | `friction-journal.md` | Haiku | One-week product-discovery experiment tooling (F8 hotkey + review script). |
| 6 | `situation-room.md` | Sonnet | SitDeck-style "NQ Situation Room" page — the shareable product wrapper for SweepStats (spec written by Fable 5, 2026-07-08). |

Planner decisions are marked `DECISION:` inside each blueprint; builders
mark gaps with `ASSUMPTION:` — scan for both before trusting a build.

Status (2026-07-08): #1 tilt-guard-v1 BUILT (by Fable 5 directly — it came back; all DoD checks passed, F9 hotkey needs its Startup shortcut added manually). #2 sweepstats-daily-pipeline BUILT (task "SweepStats Daily" registered, weekdays 18:30; dataset at 47 sessions). #4 gex-history-capture BUILT (folded into startup step 3; first real append happens next session morning). #3 sweepstats-deep-history BUILT (importer + fixture test pass; waiting only on the ~$10–30 vendor file purchase — see backtest/vendor_data/README.md). #5 friction-journal BUILT (F8 hotkey + review script; the one-week logging experiment starts when the F8 Startup shortcut is added). #1–5 ALL BUILT. #6 situation-room SPEC'D 2026-07-08 (not built) — buildable by Sonnet any time; no Fable dependency remains.
