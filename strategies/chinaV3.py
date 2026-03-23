"""
chinaV3 — Detection logic for the AMD liquidity framework.

All detection functions accept a pandas DataFrame with columns:
    open, high, low, close, volume
and a DatetimeIndex named 'datetime'.

Terminology follows CLAUDE.md exactly.

Detection philosophy — quality over quantity:
  - FBOS: only the clearest engineered sweeps qualify (significant level, decisive close back)
  - RBOS: confirmed by a full candle body close through the structural level
  - IDM POI: must be at a range extreme (within top/bottom 20% of the range), strong OB body
  - SMT: only engulfing patterns at swing extremes with momentum
  - Trend (1H/4H): HH+HL = bullish, LH+LL = bearish — only trade with trend
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SWING HIGHS / LOWS  (3-candle classic)
# ─────────────────────────────────────────────────────────────────────────────

def find_swings(df: pd.DataFrame) -> pd.DataFrame:
    """3-candle swing high/low. Returns df with swing_high and swing_low columns."""
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if df["high"].iloc[i] > df["high"].iloc[i-1] and df["high"].iloc[i] > df["high"].iloc[i+1]:
            sh[i] = True
        if df["low"].iloc[i]  < df["low"].iloc[i-1]  and df["low"].iloc[i]  < df["low"].iloc[i+1]:
            sl[i] = True
    out = df.copy()
    out["swing_high"] = sh
    out["swing_low"]  = sl
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TREND BIAS  (1H / 4H)
# ─────────────────────────────────────────────────────────────────────────────

def get_trend_bias(df_htf: pd.DataFrame, lookback: int = 6) -> pd.Series:
    """
    Determine trend bias from a higher-timeframe DataFrame (1H or 4H).
    Uses the last `lookback` swing points.

    Returns a Series aligned to df_htf.index with values:
      1  = bullish (last swing structure is HH + HL)
     -1  = bearish (last swing structure is LH + LL)
      0  = neutral / unclear
    """
    df = find_swings(df_htf)
    n  = len(df)
    bias = np.zeros(n, dtype=int)

    sh_prices = []
    sl_prices = []

    for i in range(n):
        if df["swing_high"].iloc[i]:
            sh_prices.append(df["high"].iloc[i])
        if df["swing_low"].iloc[i]:
            sl_prices.append(df["low"].iloc[i])

        if len(sh_prices) >= 2 and len(sl_prices) >= 2:
            sh2 = sh_prices[-2:]
            sl2 = sl_prices[-2:]
            bullish = sh2[-1] > sh2[-2] and sl2[-1] > sl2[-2]   # HH + HL
            bearish = sh2[-1] < sh2[-2] and sl2[-1] < sl2[-2]   # LH + LL
            if bullish:
                bias[i] = 1
            elif bearish:
                bias[i] = -1
            else:
                bias[i] = 0
        else:
            bias[i] = 0

    return pd.Series(bias, index=df_htf.index, name="trend_bias")


def align_trend_to(bias_series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill a higher-timeframe trend bias series onto a lower-timeframe index.
    Each lower-TF bar gets the bias of the last confirmed higher-TF bar.
    """
    combined = bias_series.reindex(bias_series.index.union(target_index))
    filled   = combined.ffill().fillna(0).astype(int)
    return filled.reindex(target_index)


# ─────────────────────────────────────────────────────────────────────────────
# FBOS — Failed Break of Structure  (STRICT)
# ─────────────────────────────────────────────────────────────────────────────

def find_fbos(df: pd.DataFrame, lookback: int = 6,
              min_sweep_pct: float = 0.0005) -> pd.DataFrame:
    """
    Detect FBOS: price sweeps a SIGNIFICANT swing high/low and closes back inside.

    Strict criteria:
      1. The swept level must be the most recent clean swing high/low in the lookback
      2. The wick must exceed the level by at least min_sweep_pct of price
         (filters micro-wicks that barely touch the level)
      3. The candle body must close decisively back inside (close < high of swept level
         for bearish FBOS; close > low for bullish FBOS) by at least 1 ATR(14) * 0.1
      4. Only one FBOS per swing level (once swept, level is consumed)

    Adds columns: fbos_bearish, fbos_bullish, fbos_level
    """
    df   = find_swings(df)
    atr  = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    n    = len(df)
    fbos_bear  = np.zeros(n, dtype=bool)
    fbos_bull  = np.zeros(n, dtype=bool)
    fbos_level = np.full(n, np.nan)

    used_sh: set = set()
    used_sl: set = set()

    for i in range(lookback + 14, n):
        window = df.iloc[i - lookback: i]
        candle = df.iloc[i]
        atr_val = atr.iloc[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue

        # Most recent swing high NOT already swept
        sh_window = window[window["swing_high"]]
        sh_window = sh_window[~sh_window.index.isin(used_sh)]
        if not sh_window.empty:
            ref_high = sh_window["high"].iloc[-1]
            ref_ts   = sh_window.index[-1]
            min_sweep = ref_high * min_sweep_pct
            # Wick must clearly exceed the level; close must retreat back below it
            if (candle["high"] > ref_high + min_sweep and
                    candle["close"] < ref_high - atr_val * 0.1):
                fbos_bear[i]  = True
                fbos_level[i] = ref_high
                used_sh.add(ref_ts)

        # Most recent swing low NOT already swept
        sl_window = window[window["swing_low"]]
        sl_window = sl_window[~sl_window.index.isin(used_sl)]
        if not sl_window.empty:
            ref_low = sl_window["low"].iloc[-1]
            ref_ts  = sl_window.index[-1]
            min_sweep = ref_low * min_sweep_pct
            if (candle["low"] < ref_low - min_sweep and
                    candle["close"] > ref_low + atr_val * 0.1):
                fbos_bull[i]  = True
                fbos_level[i] = ref_low
                used_sl.add(ref_ts)

    out = df.copy()
    out["fbos_bearish"] = fbos_bear
    out["fbos_bullish"] = fbos_bull
    out["fbos_level"]   = fbos_level
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RBOS — Real Break of Structure  (STRICT)
# ─────────────────────────────────────────────────────────────────────────────

def find_rbos(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Detect RBOS: full candle body closes through a structural level
    in the OPPOSITE direction after an FBOS.

    Strict criteria:
      1. Must follow an FBOS (manipulation phase must be in progress)
      2. The candle BODY (not wick) must close through the structural reference level
      3. Only the first RBOS after each FBOS counts
    """
    df = find_fbos(df, lookback=lookback)
    n  = len(df)
    rbos_bull = np.zeros(n, dtype=bool)
    rbos_bear = np.zeros(n, dtype=bool)

    last_fbos_bull_i = None
    last_fbos_bear_i = None
    bull_ref_high    = np.nan
    bear_ref_low     = np.nan

    for i in range(n):
        if df["fbos_bullish"].iloc[i]:
            last_fbos_bull_i = i
            # The RBOS target: highest swing high in the lookback before FBOS
            window = df.iloc[max(0, i - lookback * 3): i]
            sh = window[window["swing_high"]]["high"]
            bull_ref_high = sh.max() if not sh.empty else np.nan

        if df["fbos_bearish"].iloc[i]:
            last_fbos_bear_i = i
            window = df.iloc[max(0, i - lookback * 3): i]
            sl = window[window["swing_low"]]["low"]
            bear_ref_low = sl.min() if not sl.empty else np.nan

        # Bullish RBOS: candle CLOSE above the structural swing high
        # (close through the level confirms — wick alone does not count)
        if (last_fbos_bull_i is not None and i > last_fbos_bull_i
                and not np.isnan(bull_ref_high)):
            candle = df.iloc[i]
            if candle["close"] > bull_ref_high:
                rbos_bull[i]     = True
                last_fbos_bull_i = None
                bull_ref_high    = np.nan

        # Bearish RBOS: candle close below the structural swing low
        if (last_fbos_bear_i is not None and i > last_fbos_bear_i
                and not np.isnan(bear_ref_low)):
            candle = df.iloc[i]
            if candle["close"] < bear_ref_low:
                rbos_bear[i]     = True
                last_fbos_bear_i = None
                bear_ref_low     = np.nan

    out = df.copy()
    out["rbos_bullish"] = rbos_bull
    out["rbos_bearish"] = rbos_bear
    return out


# ─────────────────────────────────────────────────────────────────────────────
# IDM POI — Inducement Point of Interest  (STRICT)
# ─────────────────────────────────────────────────────────────────────────────

def find_idm_poi(df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Find IDM POIs on the 5min chart.

    Strict criteria:
      1. Must be formed at the range extreme — within top/bottom 25% of the FBOS→RBOS range
      2. The 2-candle OB must have a STRONG body (body ≥ 60% of candle range)
      3. The OB candle must be in the DIRECTION of the manipulation move
         (bearish candle for bullish POI — sellers were trapped there)
      4. The POI must NOT be mid-range (distance from range mid > 25% of range)
    """
    df_15m = find_rbos(df_15m)
    df_5m  = find_swings(df_5m.copy())
    df_5m  = df_5m.copy()

    df_5m["poi_bullish"] = False
    df_5m["poi_bearish"] = False
    df_5m["poi_high"]    = np.nan
    df_5m["poi_low"]     = np.nan

    rbos_bull_ts = df_15m.index[df_15m["rbos_bullish"]]
    rbos_bear_ts = df_15m.index[df_15m["rbos_bearish"]]
    fbos_bull_ts = df_15m.index[df_15m["fbos_bullish"]]
    fbos_bear_ts = df_15m.index[df_15m["fbos_bearish"]]

    # ── Bullish IDM POIs ──────────────────────────────────────────────────────
    for rbos_ts in rbos_bull_ts:
        prior_fbos = fbos_bull_ts[fbos_bull_ts < rbos_ts]
        if prior_fbos.empty:
            continue
        fbos_ts = prior_fbos[-1]

        window = df_5m.loc[(df_5m.index >= fbos_ts) & (df_5m.index <= rbos_ts)]
        if len(window) < 4:
            continue

        range_high = window["high"].max()
        range_low  = window["low"].min()
        range_size = range_high - range_low
        if range_size == 0:
            continue

        range_mid    = range_low + range_size * 0.5
        bottom_zone  = range_low + range_size * 0.35   # must be in bottom 35% of range

        # Find swing lows in the BOTTOM 35% of the range (induced structure)
        sl_in_window = window[window["swing_low"] & (window["low"] <= bottom_zone)]
        if sl_in_window.empty:
            continue

        poi_idx = sl_in_window["low"].idxmin()
        poi_loc = df_5m.index.get_loc(poi_idx)
        if poi_loc < 1:
            continue

        ob_candle   = df_5m.iloc[poi_loc]
        prev_candle = df_5m.iloc[poi_loc - 1]

        # OB candle must be bearish (sellers trapped at the lows, price about to reverse up)
        if ob_candle["close"] >= ob_candle["open"]:
            continue

        # OB candle must have a strong body (≥60% of candle range)
        candle_range = ob_candle["high"] - ob_candle["low"]
        if candle_range == 0:
            continue
        body = abs(ob_candle["close"] - ob_candle["open"])
        if body / candle_range < 0.40:
            continue

        poi_h = max(ob_candle["open"], prev_candle["high"])   # top of OB zone
        poi_l = ob_candle["low"]                               # bottom of OB zone

        # Must not be mid-range
        poi_mid = (poi_h + poi_l) / 2
        if abs(poi_mid - range_mid) < range_size * 0.20:
            continue

        df_5m.at[poi_idx, "poi_bullish"] = True
        df_5m.at[poi_idx, "poi_high"]    = poi_h
        df_5m.at[poi_idx, "poi_low"]     = poi_l

    # ── Bearish IDM POIs ──────────────────────────────────────────────────────
    for rbos_ts in rbos_bear_ts:
        prior_fbos = fbos_bear_ts[fbos_bear_ts < rbos_ts]
        if prior_fbos.empty:
            continue
        fbos_ts = prior_fbos[-1]

        window = df_5m.loc[(df_5m.index >= fbos_ts) & (df_5m.index <= rbos_ts)]
        if len(window) < 4:
            continue

        range_high = window["high"].max()
        range_low  = window["low"].min()
        range_size = range_high - range_low
        if range_size == 0:
            continue

        range_mid   = range_low + range_size * 0.5
        top_zone    = range_high - range_size * 0.35   # must be in top 35% of range

        sh_in_window = window[window["swing_high"] & (window["high"] >= top_zone)]
        if sh_in_window.empty:
            continue

        poi_idx = sh_in_window["high"].idxmax()
        poi_loc = df_5m.index.get_loc(poi_idx)
        if poi_loc < 1:
            continue

        ob_candle   = df_5m.iloc[poi_loc]
        prev_candle = df_5m.iloc[poi_loc - 1]

        # OB candle must be bullish (buyers trapped at the highs)
        if ob_candle["close"] <= ob_candle["open"]:
            continue

        candle_range = ob_candle["high"] - ob_candle["low"]
        if candle_range == 0:
            continue
        body = abs(ob_candle["close"] - ob_candle["open"])
        if body / candle_range < 0.40:
            continue

        poi_h = ob_candle["high"]                              # top of OB zone
        poi_l = min(ob_candle["open"], prev_candle["low"])     # bottom of OB zone

        poi_mid = (poi_h + poi_l) / 2
        if abs(poi_mid - range_mid) < range_size * 0.20:
            continue

        df_5m.at[poi_idx, "poi_bearish"] = True
        df_5m.at[poi_idx, "poi_high"]    = poi_h
        df_5m.at[poi_idx, "poi_low"]     = poi_l

    return df_5m


# ─────────────────────────────────────────────────────────────────────────────
# SMT — Smart Money Trap  (STRICT)
# ─────────────────────────────────────────────────────────────────────────────

def find_smt(df_3m: pd.DataFrame) -> pd.DataFrame:
    """
    Detect SMTs on the 3min chart — 2-candle engulfing OB at a swing extreme.

    Strict criteria:
      1. Must occur at a 3-candle swing high or swing low
      2. Engulfing body must be ≥ 80% of prior candle body (strong conviction)
      3. Prior candle must have a substantial body (≥ 50% of its range)
         — filters doji/indecision candles
      4. Current candle must close at least 50% into the prior candle's body
    """
    df = find_swings(df_3m.copy())
    n  = len(df)

    smt_bull = np.zeros(n, dtype=bool)
    smt_bear = np.zeros(n, dtype=bool)
    smt_h    = np.full(n, np.nan)
    smt_l    = np.full(n, np.nan)

    for i in range(1, n - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        prev_range = prev["high"] - prev["low"]
        curr_range = curr["high"] - curr["low"]
        if prev_range == 0 or curr_range == 0:
            continue

        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])

        # Prior candle must have substance
        if prev_body / prev_range < 0.45:
            continue

        # ── Bullish SMT: at swing low, prior bearish, current bullish engulf ──
        if df["swing_low"].iloc[i]:
            if (prev["close"] < prev["open"] and               # prior bearish
                    curr["close"] > curr["open"] and            # curr bullish
                    curr["close"] > prev["open"] and            # closes above prior open
                    curr_body >= prev_body * 0.80 and           # strong engulf
                    curr["close"] >= prev["open"] - prev_body * 0.5):  # closes deep into prior
                smt_bull[i] = True
                smt_h[i] = max(prev["high"], curr["high"])
                smt_l[i] = min(prev["low"],  curr["low"])

        # ── Bearish SMT: at swing high, prior bullish, current bearish engulf ─
        if df["swing_high"].iloc[i]:
            if (prev["close"] > prev["open"] and               # prior bullish
                    curr["close"] < curr["open"] and            # curr bearish
                    curr["close"] < prev["open"] and            # closes below prior open
                    curr_body >= prev_body * 0.80 and
                    curr["close"] <= prev["open"] + prev_body * 0.5):
                smt_bear[i] = True
                smt_h[i] = max(prev["high"], curr["high"])
                smt_l[i] = min(prev["low"],  curr["low"])

    df["smt_bullish"] = smt_bull
    df["smt_bearish"] = smt_bear
    df["smt_high"]    = smt_h
    df["smt_low"]     = smt_l
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DIV — Divergence between NQ and ES
# ─────────────────────────────────────────────────────────────────────────────

def find_div(df_nq: pd.DataFrame, df_es: pd.DataFrame,
             lookback: int = 8) -> pd.DataFrame:
    """
    Detect DIV: NQ and ES fail to confirm each other's swing extremes.

    Bullish DIV:  NQ makes a lower low  but ES does NOT → NQ likely to reverse up
    Bearish DIV:  NQ makes a higher high but ES does NOT → NQ likely to reverse down
    """
    df_nq = find_swings(df_nq.copy())
    df_es = find_swings(df_es.copy())

    common = df_nq.index.intersection(df_es.index)
    nq = df_nq.loc[common].copy()
    es = df_es.loc[common]
    n  = len(nq)

    div_bull = np.zeros(n, dtype=bool)
    div_bear = np.zeros(n, dtype=bool)

    nq_sh_hist, es_sh_hist = [], []
    nq_sl_hist, es_sl_hist = [], []

    for i in range(n):
        if nq["swing_high"].iloc[i] and es["swing_high"].iloc[i]:
            nq_sh_hist.append(nq["high"].iloc[i])
            es_sh_hist.append(es["high"].iloc[i])
            if len(nq_sh_hist) >= 2:
                if nq_sh_hist[-1] > nq_sh_hist[-2] and not (es_sh_hist[-1] > es_sh_hist[-2]):
                    div_bear[i] = True

        if nq["swing_low"].iloc[i] and es["swing_low"].iloc[i]:
            nq_sl_hist.append(nq["low"].iloc[i])
            es_sl_hist.append(es["low"].iloc[i])
            if len(nq_sl_hist) >= 2:
                if nq_sl_hist[-1] < nq_sl_hist[-2] and not (es_sl_hist[-1] < es_sl_hist[-2]):
                    div_bull[i] = True

    nq["div_bullish"] = div_bull
    nq["div_bearish"] = div_bear

    df_nq = df_nq.copy()
    df_nq["div_bullish"] = False
    df_nq["div_bearish"] = False
    df_nq.loc[common, "div_bullish"] = nq["div_bullish"]
    df_nq.loc[common, "div_bearish"] = nq["div_bearish"]
    return df_nq
