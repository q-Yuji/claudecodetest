# BLUEPRINT 3: SweepStats deep history — backfill ~12 months of NQ sessions from a purchased dataset

**BUILDER:** Claude Sonnet, working alone, cold start, cannot ask questions.
(Data normalization with timezone traps — Sonnet.)

## GOAL

The SweepStats dataset (`backtest/session_stats_dataset.json`) covers ~12
months of NQ sessions instead of ~10 weeks, ingested from a one-off
purchased intraday file, with every backfilled record tagged with its data
source so live-yfinance and purchased-history records are distinguishable.
Bigger n = credible published stats (46 sessions is a demo; ~250 is a
product).

## CONTEXT THE BUILDER NEEDS

- Files to read first: `backtest/session_stats.py` — especially the module
  docstring (v1 definitions: sessions in ET, Asia 18:00–00:00 prior day,
  London 02:00–05:00, NY 09:30–16:00; 5m bars; fakeout = reclaim close
  within 2 bars of first NY breach; DATASET_VERSION=1) and the dataset
  append/idempotency logic. Also `backtest/session_stats_dataset.json`
  (record schema — copy it exactly for backfilled records).
- Why purchased data: probed 2026-07-07 — IBKR Client Portal API cannot
  serve expired futures contracts (no continuous contract; pre-roll bars
  are ghost prints), so deep history must come from (a) TWS desktop API
  with includeExpired, or (b) a paid one-off (Databento/FirstRateData,
  ~$10–30 for a year of NQ 5m). DECISION: build for path (b) — a
  file-based importer; it has no desktop-app dependency and the file
  format is stable. The user buys the file; this build makes the importer
  ready and testable before purchase.
- Gotchas: purchased NQ data is per-contract (NQH6/NQM6/NQU6...) or
  pre-stitched continuous. Session logic needs continuous prices ONLY for
  level geometry within a single session-day, so per-day contract choice =
  the front contract by volume that day; price GAPS between contracts do
  not matter because no record spans a roll. Timestamps in vendor files
  are usually UTC or ET-without-tz — the importer must localize explicitly
  to ET and must NOT double-convert (verify with a known day: 2026-07-02
  Asia Low ≈ 29,977? no — verify against dataset records that already
  exist from yfinance; overlapping dates must classify identically).

## CONSTRAINTS

- Must stay inside: new file `backtest/import_history.py`, new folder
  `backtest/vendor_data/` (gitignored — add the line), and additions to
  `backtest/session_stats_dataset.json` records only.
- Must not change: v1 definitions, DATASET_VERSION, existing dataset
  records (append/backfill only — never rewrite a date that already
  exists from the live source).
- Stack: pandas (already used). No new pip installs.
- Non-negotiables: every backfilled record gets `"source": "vendor"` added;
  existing records are treated as `source: "live"` implicitly. Overlap
  dates (present in both) keep the live record and log the comparison.

## STEP-BY-STEP PLAN

1. Read `backtest/session_stats.py` and identify the function that turns
   one day's 5m DataFrame into a dataset record (session ranges +
   first-touch classification). If classification is inline, extract
   NOTHING — instead import and reuse via the module's public functions;
   only if truly impossible, copy the minimal logic into the importer with
   a comment naming the source lines.
2. Create `backtest/import_history.py` with CLI:
   `python -m backtest.import_history <csv_path> [--tz utc|et] [--dry-run]`
   a. Load vendor CSV; accept the two common shapes:
      `timestamp,open,high,low,close,volume` (single continuous series) or
      `date,time,open,high,low,close,volume`. Auto-detect by header.
   b. Localize timestamps per `--tz` (default `utc`) → convert to ET.
   c. Resample/verify bars are 5m; if 1m, resample to 5m (label=left,
      closed=left — match yfinance convention used by the engine).
   d. For each complete session-day not already in the dataset: build the
      record with the SAME code path as the live engine, add
      `"source": "vendor"`, append.
   e. For overlap days: classify, compare to the existing live record,
      print MATCH/MISMATCH per level, change nothing.
   f. `--dry-run` prints what would be appended without writing.
3. Sort the dataset file by date after appending; write with the same
   json formatting the engine uses (open the file to match indent).
4. Create a synthetic-fixture test `backtest/test_import_history.py`
   (plain `python backtest/test_import_history.py`, no pytest dependency):
   generate 3 fake days of 5m bars with a known Asia high fakeout on day 2,
   run the importer against a temp dataset copy, assert the record for
   day 2 says fakeout at asia_high and `source == "vendor"`.
5. Write `backtest/vendor_data/README.md`: what file to buy (Databento
   GLBX.MDP3 NQ 5m OHLCV ~12 months, or FirstRateData NQ 1m bundle),
   expected CSV shape, and the exact import command to run after download.

## EXACT INPUTS TO USE

- Kickoff prompt: "Open blueprints/sweepstats-deep-history.md and build it
  exactly as written. Reuse backtest/session_stats.py's classification code
  path; do not fork the definitions."

## DEFINITION OF DONE

- [ ] `python backtest/test_import_history.py` exits 0 (fixture day 2
      classified fakeout/asia_high, tagged vendor).
- [ ] `python -m backtest.import_history --help` documents csv path, --tz,
      --dry-run.
- [ ] Dry-run against a small real vendor sample (or the fixture) prints
      per-day planned appends and touches nothing.
- [ ] Overlap handling proven: importing a fixture that includes an
      existing live date changes 0 existing records and prints a
      MATCH/MISMATCH line.
- [ ] `backtest/vendor_data/` is gitignored; README.md inside it tells the
      user exactly what to purchase and the one command to run.

## IF SOMETHING IS UNCLEAR

Smallest safe assumption, `ASSUMPTION:` tag, keep going. If the engine's
internals genuinely can't be reused without refactoring, stop at the
smallest copied function and flag it — do not refactor session_stats.py.
