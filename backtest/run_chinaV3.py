"""
chinaV3 backtest — NQ only, intraday day-trader approach.

AMD context on 15min, entries on 3min SMT (2-candle orderblock):

Timeframes:
  4H    → confirms 1H bias (high conviction only — not required)
  1H    → primary trend bias for the session
  15min → AMD context: FBOS/RBOS detection → IDM TARGET levels (not entries)
  3min  → SMT entries (2-candle OB at swing extreme)
  1min  → OTE entry refinement (optional)
  ES    → DIV filter only

Entry mechanics:
  SMT = 2-candle orderblock on 3min at swing extreme.
  Entry at CE (50% midpoint of SMT range).
  Stop beyond far edge of 3min range.
  TP: nearest IDM target (15min) ≥50pts away; fallback to 1H swing.

Session trading windows (UTC):
  Asia   : 00:00–08:00
  NY AM  : 13:30–18:00

Sizing: 2 contracts standard, 3 on high-conviction (4H + 1H agree).
Max stop: 10 NQ points.  Min target: 50 NQ points.

Prop firm: MyFutureFunded / Topstep $50k eval rules.
  Eval:   profit target $3,000 | daily loss $1,500 | trailing DD $2,000
  Funded: daily loss $1,500    | trailing DD $2,000 (floor = starting balance)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.chinaV3 import (
    find_rbos, find_div, get_trend_bias, align_trend_to, find_swings, find_smt
)
from backtest.engine import PropFirmEngine

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ASSET      = "NQ"
CORR_ASSET = "ES"
TICK       = 0.25
MIN_POI_W  = 4.0   # minimum 4 NQ points (intraday POIs are tighter than swing POIs)


def load(asset: str, tf: str) -> pd.DataFrame:
    path = DATA_DIR / asset / f"{tf}.csv"
    df = pd.read_csv(path, parse_dates=True)

    # Handle TradingView manual export format (time column) vs fetched format (datetime index)
    if "time" in df.columns:
        df = df.rename(columns={"time": "datetime"})
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    return df.sort_index()


OTE_FIB = 0.62   # enter at 62% into the exhaustion candle's body


def _find_ote(bars_1m: pd.DataFrame, direction: str,
              poi_h: float, poi_l: float) -> float | None:
    """
    Scan 1min bars inside the POI zone for an exhaustion candle and return the
    OTE entry price (62% into the candle body from open).

    Exhaustion candle criteria:
      - Must touch or overlap the POI zone
      - Body ≥ 55% of candle range  (impulsive, not doji)
      - Body > 1.3 × median body of the window  (larger than recent average)
      - Directional: bullish candle for long setup, bearish for short

    Returns None if no qualifying candle found (caller uses CE instead).
    """
    if bars_1m.empty:
        return None

    bodies = (bars_1m["close"] - bars_1m["open"]).abs()
    median_body = bodies.median()

    for _, c in bars_1m.iterrows():
        rng  = c["high"] - c["low"]
        body = abs(c["close"] - c["open"])
        if rng == 0 or body == 0:
            continue

        # Candle must touch or overlap the POI zone
        if c["high"] < poi_l or c["low"] > poi_h:
            continue

        # Impulsive body filter
        if body / rng < 0.55:
            continue
        if median_body > 0 and body < 1.3 * median_body:
            continue

        if direction == "long" and c["close"] > c["open"]:
            # Bullish exhaustion inside POI — buyers stepping in
            ote = c["open"] + OTE_FIB * (c["close"] - c["open"])
            return round(ote, 2)

        if direction == "short" and c["close"] < c["open"]:
            # Bearish exhaustion inside POI — sellers stepping in
            ote = c["open"] - OTE_FIB * (c["open"] - c["close"])
            return round(ote, 2)

    return None


def run_backtest() -> dict:
    print(f"\n{'='*58}")
    print(f"  chinaV3 Backtest — {ASSET}  (MyFutureFunded $50k)")
    print(f"  Intraday day-trader mode — 5min AMD cycle detection")
    print(f"{'='*58}")

    df_4h   = load(ASSET,      "4h")
    df_1h   = load(ASSET,      "1h")
    df_5    = load(ASSET,      "5min")
    df_5_es = load(CORR_ASSET, "5min")

    # 3min: resample from 5min for full 60-day coverage
    # (yfinance 1min only gives 7 days; resampling 5min → 3min gives 60 days of SMT data)
    df_3 = df_5.resample("3min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last",  "volume": "sum"
    }).dropna()

    try:
        df_1 = load(ASSET, "1min")
    except FileNotFoundError:
        df_1 = None

    print(f"  5min data  : {df_5.index[0].date()}  →  {df_5.index[-1].date()}")
    print(f"  4H bars: {len(df_4h)}  1H bars: {len(df_1h)}  "
          f"5min: {len(df_5)}  3min: {len(df_3)}  "
          f"1min: {len(df_1) if df_1 is not None else 0}")

    # ── AMD context on 15min ─────────────────────────────────────────────────
    # 15min: read the AMD cycle, identify FBOS/RBOS, build IDM POI TARGET levels.
    # 15min POIs are NOT entries — they are the distribution targets for 3min SMTs.
    print("  Detecting 15min AMD structure...")
    df_15 = load(ASSET, "15min")
    df_15 = find_rbos(df_15)
    df_15 = find_swings(df_15)

    n_fbos_15 = int(df_15['fbos_bullish'].sum() + df_15['fbos_bearish'].sum())
    n_rbos_15 = int(df_15['rbos_bullish'].sum() + df_15['rbos_bearish'].sum())
    print(f"  15min FBOS: {n_fbos_15}   RBOS: {n_rbos_15}")

    # Build IDM POI levels from 15min — these are TARGETS, not entry zones
    idm_targets = []   # {"level": float, "direction": "long"/"short", "formed": ts}
    last_fbos_bull_15 = None
    last_fbos_bear_15 = None

    for ts, row in df_15.iterrows():
        if row["fbos_bullish"]: last_fbos_bull_15 = (ts, row.copy())
        if row["fbos_bearish"]: last_fbos_bear_15 = (ts, row.copy())

        if row["rbos_bullish"] and last_fbos_bull_15 is not None:
            _, fr = last_fbos_bull_15
            level = max(fr["open"], fr["close"])  # top of 15min FBOS body = IDM target
            if (max(fr["open"], fr["close"]) - min(fr["open"], fr["close"])) >= MIN_POI_W:
                idm_targets.append({"formed": ts, "direction": "long", "level": level})
            last_fbos_bull_15 = None

        if row["rbos_bearish"] and last_fbos_bear_15 is not None:
            _, fr = last_fbos_bear_15
            level = min(fr["open"], fr["close"])  # bottom of 15min FBOS body = IDM target
            if (max(fr["open"], fr["close"]) - min(fr["open"], fr["close"])) >= MIN_POI_W:
                idm_targets.append({"formed": ts, "direction": "short", "level": level})
            last_fbos_bear_15 = None

    print(f"  15min IDM target levels: {len(idm_targets)}")

    # DIV on 5min (ES comparison)
    df_5 = find_div(df_5, df_5_es)
    df_5 = find_swings(df_5)

    # ── 3min SMT detection — primary entries ─────────────────────────────────
    # SMT = 2-candle orderblock on 3min drawn at a swing extreme.
    # Entry: CE of the 2-candle range. Stop: beyond far edge.
    # Valid only when there is a higher IDM POI target above (long) or below (short).
    smt_list = []
    if df_3 is not None:
        df_3 = find_smt(df_3)
        n_smt = int(df_3['smt_bullish'].sum() + df_3['smt_bearish'].sum())
        print(f"  3min SMTs : {n_smt}")
        for ts_3, row_3 in df_3.iterrows():
            if row_3["smt_bullish"]:
                smt_list.append({
                    "formed": ts_3, "direction": "long",
                    "range_high": row_3["smt_high"], "range_low": row_3["smt_low"],
                    "active": True
                })
            elif row_3["smt_bearish"]:
                smt_list.append({
                    "formed": ts_3, "direction": "short",
                    "range_high": row_3["smt_high"], "range_low": row_3["smt_low"],
                    "active": True
                })
    else:
        print("  3min data : not available")

    # ── trend bias: 1H primary, 4H for high-conviction sizing ────────────────
    bias_4h    = get_trend_bias(df_4h)
    bias_1h    = get_trend_bias(df_1h)
    bias_4h_5m = align_trend_to(bias_4h, df_5.index)
    bias_1h_5m = align_trend_to(bias_1h, df_5.index)

    primary_bias = bias_1h_5m
    hc_bias = pd.Series(
        np.where((bias_4h_5m == 1)  & (bias_1h_5m == 1),   1,
        np.where((bias_4h_5m == -1) & (bias_1h_5m == -1), -1, 0)),
        index=df_5.index
    )
    print(f"  1H bias — Bull: {int((primary_bias==1).sum())}  "
          f"Bear: {int((primary_bias==-1).sum())}  "
          f"Neutral: {int((primary_bias==0).sum())}")
    print(f"  High-conviction (4H+1H) — Bull: {int((hc_bias==1).sum())}  "
          f"Bear: {int((hc_bias==-1).sum())}")

    # TP: 15min IDM level OR 1H swing at 50pt+ — whichever is closer but still ≥50pts
    df_1h_sw = find_swings(df_1h.copy())
    sh_tp = df_1h_sw[df_1h_sw["swing_high"]][["high"]]
    sl_tp = df_1h_sw[df_1h_sw["swing_low"]][["low"]]
    MIN_TARGET_PTS = 50.0

    # ── simulate on 5min bars ─────────────────────────────────────────────────
    engine = PropFirmEngine()
    print(f"\n  Running simulation...\n")

    _prev_date: date | None = None

    for ts, candle in df_5.iterrows():

        if not engine.active:
            break

        # ── EOD update: call end_of_day when the date rolls over ─────────────
        today = ts.date()
        if _prev_date is not None and today != _prev_date:
            engine.end_of_day(_prev_date)
        _prev_date = today

        engine.check_exit(candle, ts)

        if engine._open_trade is not None:
            continue

        # ── Session filter: Asia (00:00–08:00 UTC) + NY AM (13:30–18:00 UTC) ──
        h = ts.hour
        in_asia = 0 <= h < 8
        in_nyam = 13 <= h < 18
        if not (in_asia or in_nyam):
            continue

        trend    = primary_bias.get(ts, 0)
        hc       = hc_bias.get(ts, 0)          # ±1 if 4H+1H agree, else 0
        div_bull = bool(df_5.loc[ts, "div_bullish"])
        div_bear = bool(df_5.loc[ts, "div_bearish"])

        # ── SMT entries (3min 2-candle OB) — primary and only entry mechanism ─
        for smt in smt_list:
            if not smt["active"] or ts <= smt["formed"]:
                continue

            rng_h = smt["range_high"]
            rng_l = smt["range_low"]
            rng_w = rng_h - rng_l

            # Expire SMT when close > 50% of range past far edge
            if smt["direction"] == "long"  and candle["close"] < rng_l - rng_w * 0.5:
                smt["active"] = False
                continue
            if smt["direction"] == "short" and candle["close"] > rng_h + rng_w * 0.5:
                smt["active"] = False
                continue

            # Trend filter (1H primary)
            if smt["direction"] == "long"  and trend == -1:
                continue
            if smt["direction"] == "short" and trend == 1:
                continue

            # CE = 50% midpoint (reference only — entry triggers on first zone touch)
            ce = round((rng_h + rng_l) / 2, 2)

            if smt["direction"] == "long":
                # Trigger: any candle that touches/overlaps the SMT zone
                if not (candle["low"] <= rng_h and candle["high"] >= rng_l):
                    continue
                if div_bear:            # ES DIV bearish → skip long
                    continue

                # Entry at CE — triggers on any zone touch, not requiring exact CE candle
                entry = ce
                stop  = rng_l - 2 * TICK

                # TP: nearest IDM target (15min) ≥50pts above — then 1H swing fallback
                target = None
                for idt in idm_targets:
                    if (idt["formed"] < ts and
                            idt["direction"] == "long" and
                            idt["level"] > entry + MIN_TARGET_PTS):
                        if target is None or idt["level"] < target:
                            target = idt["level"]

                if target is None:
                    cands = sh_tp[sh_tp.index < ts]
                    cands = cands[cands["high"] > entry + MIN_TARGET_PTS]
                    if cands.empty:
                        continue
                    target = float(cands["high"].iloc[-1])

                hc_trade = bool(hc == 1)
                t = engine.open_trade(ASSET, "long", entry, stop, target, ts,
                                      div_confirmed=bool(div_bull),
                                      high_conviction=hc_trade,
                                      entry_type="SMT")
                if t:
                    smt["active"] = False
                    print(f"  [{ts}] SMT LONG   entry={t.entry_price:.2f} (CE={ce:.2f})  "
                          f"sl={stop:.2f}  tp={target:.2f}  RR={t.rr}  "
                          f"w={rng_w:.1f}pts  HC={hc_trade}")
                    break

            elif smt["direction"] == "short":
                if not (candle["high"] >= rng_l and candle["low"] <= rng_h):
                    continue
                if div_bull:            # ES DIV bullish → skip short
                    continue

                # Entry at CE — triggers on any zone touch, not requiring exact CE candle
                entry = ce
                stop  = rng_h + 2 * TICK

                # TP: nearest IDM target (15min) ≥50pts below — then 1H swing fallback
                target = None
                for idt in idm_targets:
                    if (idt["formed"] < ts and
                            idt["direction"] == "short" and
                            idt["level"] < entry - MIN_TARGET_PTS):
                        if target is None or idt["level"] > target:
                            target = idt["level"]

                if target is None:
                    cands = sl_tp[sl_tp.index < ts]
                    cands = cands[cands["low"] < entry - MIN_TARGET_PTS]
                    if cands.empty:
                        continue
                    target = float(cands["low"].iloc[-1])

                hc_trade = bool(hc == -1)
                t = engine.open_trade(ASSET, "short", entry, stop, target, ts,
                                      div_confirmed=bool(div_bear),
                                      high_conviction=hc_trade,
                                      entry_type="SMT")
                if t:
                    smt["active"] = False
                    print(f"  [{ts}] SMT SHORT  entry={t.entry_price:.2f} (CE={ce:.2f})  "
                          f"sl={stop:.2f}  tp={target:.2f}  RR={t.rr}  "
                          f"w={rng_w:.1f}pts  HC={hc_trade}")
                    break

    # Close any open trade and run final EOD at end of data
    if engine._open_trade is not None:
        last = df_5.iloc[-1]
        engine.close_trade(float(last["close"]), df_5.index[-1])
    if _prev_date is not None:
        engine.end_of_day(_prev_date)

    # ── results ───────────────────────────────────────────────────────────────
    r = engine.result.summarise().to_dict()
    phase = r.get("prop_phase", "eval")

    all_trades = r.get("trades", [])

    print(f"\n{'─'*58}")
    print(f"  Prop firm : {phase.upper()}"
          + (f"  ({r['prop_fail_reason']})" if r.get('prop_fail_reason') else ""))
    print(f"  Trades    : {r['total_trades']}  "
          f"(W: {r['wins']}  L: {r['losses']})")
    print(f"  Win rate  : {r['win_rate']}%")
    print(f"  Avg R:R   : {r['avg_rr']}R")
    print(f"  P-factor  : {r['profit_factor']}")
    print(f"  Net P&L   : ${r['net_pnl']:,.2f}")
    print(f"  Max DD    : ${r['max_drawdown']:,.2f}  ({r['max_drawdown_pct']}%)")
    print(f"  Balance   : ${r['ending_balance']:,.2f}  (started $50,000)")
    print(f"  Winning days (funded): {r.get('winning_days', 0)} / 5 needed")
    if r.get('payout_eligible'):
        print(f"  Total withdrawn: ${r.get('total_withdrawn', 0):,.2f}")
    print(f"{'─'*58}\n")

    return r


def main():
    result = run_backtest()
    out_path = RESULTS_DIR / "chinaV3_results.json"
    with open(out_path, "w") as f:
        json.dump({"NQ": result}, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
