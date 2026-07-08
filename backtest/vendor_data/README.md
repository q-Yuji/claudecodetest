# Vendor data drop folder

Put the purchased NQ intraday history file here. Everything in this folder
except this README is gitignored — the durable output is the records the
importer appends to `backtest/session_stats_dataset.json`, not the raw file.

## What to buy (one-off, ~$10–30)

Either of:

1. **Databento** — dataset `GLBX.MDP3`, schema `ohlcv-1m` (or `ohlcv-5m`),
   symbols `NQ` (parent, all contracts) or the individual contracts covering
   the last ~12 months, CSV export. Timestamps are UTC (`ts_event`).
2. **FirstRateData** — "NQ (E-mini Nasdaq 100) 1-minute" bundle, CSV.
   Timestamps are ET without timezone — import with `--tz et`.

## Expected CSV shapes (auto-detected)

- `timestamp,open,high,low,close,volume` — single continuous series
- `ts_event,...,open,high,low,close,volume[,symbol]` — Databento; a
  `symbol`/`contract` column triggers per-session-day front-contract
  selection by volume (records never span a roll)
- `date,time,open,high,low,close,volume` — FirstRateData style

1-minute files are resampled to 5m automatically.

## How to import

```
# preview — writes nothing, prints planned appends + overlap MATCH/MISMATCH
python -m backtest.import_history backtest/vendor_data/<file>.csv --tz utc --dry-run

# real import (use --tz et for FirstRateData naive-ET files)
python -m backtest.import_history backtest/vendor_data/<file>.csv --tz utc
```

Overlap days (already in the dataset from the live source) are never
rewritten — the importer prints a per-level MATCH/MISMATCH comparison
instead. **If you see MISMATCH lines on overlap days, suspect a timezone
error and stop.** After a good import, commit
`backtest/session_stats_dataset.json` and rerun
`python -m backtest.daily_stats_run` to refresh the card.
