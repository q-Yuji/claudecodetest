"""
chinaV3 backtest — NQ only, intraday day-trader approach.

The AMD cycle plays out every day. This backtest reflects that:

  4H / 1H  → overall trend bias (must agree — both bull or both bear)
  15min     → macro structure context only (not used for signals)
  5min      → FBOS + RBOS detection (intraday manipulation sweeps)
  5min      → IDM POI formation + entry execution
  3min      → SMT detection — stop at far end of 2-candle manipulation range
  ES 5min   → DIV confirmation filter

Entry logic (IDM POI):
  1. 4H + 1H bias must agree
  2. FBOS on 5min → RBOS confirms → IDM POI = FBOS candle body (min 4pts wide)
  3. Price retests POI → entry at near edge, SL 2 ticks beyond far edge
  4. TP: nearest swing high/low before FBOS (min 8pts away)
  5. ES DIV opposing → skip

Entry logic (SMT):
  1. 4H + 1H bias must agree
  2. Bullish/bearish SMT detected on 3min at swing extreme
  3. Entry: near end of 2-candle range, SL at far end of range (only if ≤ 10pts)
  4. TP: nearest swing high/low (min 8pts away)
  5. ES DIV opposing → skip

Sizing: 2 contracts standard, 3 on high-conviction (not currently flagged).
Max stop: 10 NQ points — trade rejected if wider.

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
    df = pd.read_csv(DATA_DIR / asset / f"{tf}.csv",
                     index_col="datetime", parse_dates=True)
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

    # 3min for SMT entries; 1min for OTE entry refinement (both optional)
    try:
        df_3 = load(ASSET, "3min")
    except FileNotFoundError:
        df_3 = None

    try:
        df_1 = load(ASSET, "1min")
    except FileNotFoundError:
        df_1 = None

    print(f"  5min data  : {df_5.index[0].date()}  →  {df_5.index[-1].date()}")
    print(f"  4H bars: {len(df_4h)}  1H bars: {len(df_1h)}  "
          f"5min: {len(df_5)}  3min: {len(df_3) if df_3 is not None else 0}  "
          f"1min: {len(df_1) if df_1 is not None else 0}")

    # ── signal detection on 5min ──────────────────────────────────────────────
    print("  Detecting signals...")
    df_5 = find_rbos(df_5)            # also runs find_fbos internally
    df_5 = find_div(df_5, df_5_es)
    df_5 = find_swings(df_5)

    n_fbos = int(df_5['fbos_bullish'].sum() + df_5['fbos_bearish'].sum())
    n_rbos = int(df_5['rbos_bullish'].sum() + df_5['rbos_bearish'].sum())
    print(f"  5min FBOS: {n_fbos}   RBOS: {n_rbos}")

    # ── SMT detection on 3min ─────────────────────────────────────────────────
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
        print("  3min data : not available — SMT entries skipped")

    # ── trend bias: 4H + 1H must agree ───────────────────────────────────────
    bias_4h    = get_trend_bias(df_4h)
    bias_1h    = get_trend_bias(df_1h)
    bias_4h_5m = align_trend_to(bias_4h, df_5.index)
    bias_1h_5m = align_trend_to(bias_1h, df_5.index)

    combined_bias = pd.Series(
        np.where((bias_4h_5m == 1)  & (bias_1h_5m == 1),   1,
        np.where((bias_4h_5m == -1) & (bias_1h_5m == -1), -1, 0)),
        index=df_5.index
    )
    print(f"  Trend — Bull bars: {int((combined_bias==1).sum())}  "
          f"Bear: {int((combined_bias==-1).sum())}  "
          f"Neutral: {int((combined_bias==0).sum())}")

    # ── build IDM POI list ────────────────────────────────────────────────────
    # IDM POI = body of the FBOS candle, valid only after the RBOS confirms.
    # This is the manipulation trap zone. Entry when price retests it.
    poi_list       = []
    last_fbos_bull = None
    last_fbos_bear = None

    for ts, row in df_5.iterrows():
        if row["fbos_bullish"]:
            last_fbos_bull = (ts, row.copy())
        if row["fbos_bearish"]:
            last_fbos_bear = (ts, row.copy())

        if row["rbos_bullish"] and last_fbos_bull is not None:
            _, fr = last_fbos_bull
            poi_h = max(fr["open"], fr["close"])
            poi_l = min(fr["open"], fr["close"])
            if poi_h - poi_l >= MIN_POI_W:
                poi_list.append({"formed": ts, "direction": "long",
                                  "poi_high": poi_h, "poi_low": poi_l,
                                  "active": True, "session": ts.date()})
            last_fbos_bull = None

        if row["rbos_bearish"] and last_fbos_bear is not None:
            _, fr = last_fbos_bear
            poi_h = max(fr["open"], fr["close"])
            poi_l = min(fr["open"], fr["close"])
            if poi_h - poi_l >= MIN_POI_W:
                poi_list.append({"formed": ts, "direction": "short",
                                  "poi_high": poi_h, "poi_low": poi_l,
                                  "active": True, "session": ts.date()})
            last_fbos_bear = None

    print(f"  IDM POIs (after RBOS): {len(poi_list)}")

    # Swing levels for TP targets
    sh_5 = df_5[df_5["swing_high"]][["high"]]
    sl_5 = df_5[df_5["swing_low"]][["low"]]

    # ── simulate on 5min bars ─────────────────────────────────────────────────
    engine = PropFirmEngine()
    print(f"\n  Running simulation...\n")

    for ts, candle in df_5.iterrows():

        if not engine.active:
            break

        engine.check_exit(candle, ts)

        if engine._open_trade is not None:
            continue

        trend    = combined_bias.get(ts, 0)
        div_bull = bool(df_5.loc[ts, "div_bullish"])
        div_bear = bool(df_5.loc[ts, "div_bearish"])

        for poi in poi_list:
            if not poi["active"] or ts <= poi["formed"]:
                continue

            poi_h = poi["poi_high"]
            poi_l = poi["poi_low"]

            # Expire POI if price closes decisively through it
            if poi["direction"] == "long"  and candle["close"] < poi_l - 2 * TICK:
                poi["active"] = False
                continue
            if poi["direction"] == "short" and candle["close"] > poi_h + 2 * TICK:
                poi["active"] = False
                continue

            # ── Trend filter ──────────────────────────────────────────────────
            if poi["direction"] == "long"  and trend == -1:
                continue
            if poi["direction"] == "short" and trend == 1:
                continue

            # ── Long setup ────────────────────────────────────────────────────
            if poi["direction"] == "long":
                if not (candle["low"] <= poi_h and candle["high"] >= poi_l):
                    continue
                if div_bear:           # ES DIV bearish = don't go long
                    continue

                stop = poi_l - 2 * TICK
                ce   = round((poi_h + poi_l) / 2, 2)   # consequent encroachment

                # OTE from 1min exhaustion candle (if data available for this window)
                ote = None
                if df_1 is not None:
                    win_start = ts - pd.Timedelta("4min")
                    bars_1m   = df_1.loc[win_start:ts] if win_start in df_1.index or ts in df_1.index else pd.DataFrame()
                    try:
                        bars_1m = df_1.loc[win_start:ts]
                    except KeyError:
                        bars_1m = pd.DataFrame()
                    ote = _find_ote(bars_1m, "long", poi_h, poi_l)

                # Priority: OTE > CE; OTE must be above stop, CE must be above stop
                if ote is not None and ote > stop:
                    entry       = ote
                    entry_label = f"OTE={ote:.2f}"
                elif ce > stop:
                    entry       = ce
                    entry_label = "CE"
                else:
                    entry       = poi_h
                    entry_label = "POI_EDGE"

                # TP = nearest swing HIGH above entry before the FBOS
                candidates = sh_5.loc[sh_5.index < poi["formed"]]
                candidates = candidates[candidates["high"] > entry + 8]
                target = float(candidates["high"].iloc[-1]) if not candidates.empty \
                         else entry + 3 * (entry - stop)

                # POI blown this candle → instant stop
                if candle["close"] < poi_l:
                    t = engine.open_trade(ASSET, "long", entry, stop, target, ts,
                                          div_confirmed=bool(div_bull),
                                          entry_type="IDM_POI")
                    if t:
                        engine.close_trade(poi_l - TICK, ts)
                        poi["active"] = False
                        print(f"  [{ts}] LONG  BLOWN  {entry_label}  entry={t.entry_price:.2f}  "
                              f"exit={poi_l-TICK:.2f}  PnL=${t.pnl:,.0f}")
                    break

                t = engine.open_trade(ASSET, "long", entry, stop, target, ts,
                                      div_confirmed=bool(div_bull),
                                      entry_type="IDM_POI")
                if t:
                    poi["active"] = False
                    print(f"  [{ts}] LONG   {entry_label}  entry={t.entry_price:.2f}  "
                          f"sl={stop:.2f}  tp={target:.2f}  "
                          f"RR={t.rr}  DIV={bool(div_bull)}")
                    break

            # ── Short setup ───────────────────────────────────────────────────
            elif poi["direction"] == "short":
                if not (candle["high"] >= poi_l and candle["low"] <= poi_h):
                    continue
                if div_bull:           # ES DIV bullish = don't go short
                    continue

                stop = poi_h + 2 * TICK
                ce   = round((poi_h + poi_l) / 2, 2)

                ote = None
                if df_1 is not None:
                    win_start = ts - pd.Timedelta("4min")
                    try:
                        bars_1m = df_1.loc[win_start:ts]
                    except KeyError:
                        bars_1m = pd.DataFrame()
                    ote = _find_ote(bars_1m, "short", poi_h, poi_l)

                if ote is not None and ote < stop:
                    entry       = ote
                    entry_label = f"OTE={ote:.2f}"
                elif ce < stop:
                    entry       = ce
                    entry_label = "CE"
                else:
                    entry       = poi_l
                    entry_label = "POI_EDGE"

                candidates = sl_5.loc[sl_5.index < poi["formed"]]
                candidates = candidates[candidates["low"] < entry - 8]
                target = float(candidates["low"].iloc[-1]) if not candidates.empty \
                         else entry - 3 * (stop - entry)

                if candle["close"] > poi_h:
                    t = engine.open_trade(ASSET, "short", entry, stop, target, ts,
                                          div_confirmed=bool(div_bear),
                                          entry_type="IDM_POI")
                    if t:
                        engine.close_trade(poi_h + TICK, ts)
                        poi["active"] = False
                        print(f"  [{ts}] SHORT BLOWN  {entry_label}  entry={t.entry_price:.2f}  "
                              f"exit={poi_h+TICK:.2f}  PnL=${t.pnl:,.0f}")
                    break

                t = engine.open_trade(ASSET, "short", entry, stop, target, ts,
                                      div_confirmed=bool(div_bear),
                                      entry_type="IDM_POI")
                if t:
                    poi["active"] = False
                    print(f"  [{ts}] SHORT  {entry_label}  entry={t.entry_price:.2f}  "
                          f"sl={stop:.2f}  tp={target:.2f}  "
                          f"RR={t.rr}  DIV={bool(div_bear)}")
                    break

        # ── SMT entries (3min) — only if no trade opened from POI ─────────────
        if engine._open_trade is not None:
            continue

        for smt in smt_list:
            if not smt["active"] or ts <= smt["formed"]:
                continue

            rng_h = smt["range_high"]
            rng_l = smt["range_low"]
            rng_w = rng_h - rng_l

            # Expire SMT if price closes through the range far side
            if smt["direction"] == "long"  and candle["close"] < rng_l - 2 * TICK:
                smt["active"] = False
                continue
            if smt["direction"] == "short" and candle["close"] > rng_h + 2 * TICK:
                smt["active"] = False
                continue

            # Trend filter
            if smt["direction"] == "long"  and trend == -1:
                continue
            if smt["direction"] == "short" and trend == 1:
                continue

            if smt["direction"] == "long":
                if not (candle["low"] <= rng_h and candle["high"] >= rng_l):
                    continue
                if div_bear:
                    continue

                entry  = rng_h                   # near edge of SMT range
                stop   = rng_l - 2 * TICK        # far edge of range + 2 ticks

                candidates = sh_5.loc[sh_5.index < smt["formed"]]
                candidates = candidates[candidates["high"] > entry + 8]
                target = float(candidates["high"].iloc[-1]) if not candidates.empty \
                         else entry + 3 * (entry - stop)

                t = engine.open_trade(ASSET, "long", entry, stop, target, ts,
                                      div_confirmed=bool(div_bull),
                                      entry_type="SMT")
                if t:
                    smt["active"] = False
                    print(f"  [{ts}] SMT LONG   entry={t.entry_price:.2f}  "
                          f"sl={stop:.2f}  tp={target:.2f}  "
                          f"RR={t.rr}  range_w={rng_w:.1f}pts")
                    break

            elif smt["direction"] == "short":
                if not (candle["high"] >= rng_l and candle["low"] <= rng_h):
                    continue
                if div_bull:
                    continue

                entry  = rng_l                   # near edge of SMT range
                stop   = rng_h + 2 * TICK        # far edge of range + 2 ticks

                candidates = sl_5.loc[sl_5.index < smt["formed"]]
                candidates = candidates[candidates["low"] < entry - 8]
                target = float(candidates["low"].iloc[-1]) if not candidates.empty \
                         else entry - 3 * (stop - entry)

                t = engine.open_trade(ASSET, "short", entry, stop, target, ts,
                                      div_confirmed=bool(div_bear),
                                      entry_type="SMT")
                if t:
                    smt["active"] = False
                    print(f"  [{ts}] SMT SHORT  entry={t.entry_price:.2f}  "
                          f"sl={stop:.2f}  tp={target:.2f}  "
                          f"RR={t.rr}  range_w={rng_w:.1f}pts")
                    break

    # Close any open trade at end of data
    if engine._open_trade is not None:
        last = df_5.iloc[-1]
        engine.close_trade(float(last["close"]), df_5.index[-1])

    # ── results ───────────────────────────────────────────────────────────────
    r = engine.result.summarise().to_dict()
    phase = r.get("prop_phase", "eval")

    # Entry-type breakdown
    all_trades = r.get("trades", [])
    poi_trades  = [t for t in all_trades if t.get("entry_type") == "IDM_POI"]
    smt_trades  = [t for t in all_trades if t.get("entry_type") == "SMT"]

    print(f"\n{'─'*58}")
    print(f"  Prop firm : {phase.upper()}"
          + (f"  ({r['prop_fail_reason']})" if r.get('prop_fail_reason') else ""))
    print(f"  Trades    : {r['total_trades']}  "
          f"(W: {r['wins']}  L: {r['losses']})")
    print(f"  IDM POI   : {len(poi_trades)}  |  SMT: {len(smt_trades)}")
    print(f"  Win rate  : {r['win_rate']}%")
    print(f"  Avg R:R   : {r['avg_rr']}R")
    print(f"  P-factor  : {r['profit_factor']}")
    print(f"  Net P&L   : ${r['net_pnl']:,.2f}")
    print(f"  Max DD    : ${r['max_drawdown']:,.2f}  ({r['max_drawdown_pct']}%)")
    print(f"  Balance   : ${r['ending_balance']:,.2f}  (started $50,000)")
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
