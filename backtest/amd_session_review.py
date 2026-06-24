"""
AMD Session Review — 2-week day-by-day analysis of NQ.

For each trading day:
  - Asia H/L  (6:00 PM – 12:00 AM ET, prior evening)
  - London H/L (2:00 AM – 5:00 AM ET)
  - NY AM sweep / reversal detection
  - FBOS / RBOS on 15m
  - SMT on 1m (resampled to 3m)
  - DIV (NQ vs ES) on 5m
  - Big candle → ForexFactory news flag
  - GEX level placeholder (populated manually or via TV replay)

Run:
  python -m backtest.amd_session_review
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

from strategies.chinaV3 import (
    find_fbos, find_rbos, find_swings, find_smt, find_div
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ET = ZoneInfo("America/New_York")

# ── Session windows (ET) ───────────────────────────────────────────────────────
# Asia  : 6:00 PM – 12:00 AM ET  (prior evening relative to trading day)
# London: 2:00 AM –  5:00 AM ET
# NY AM : 9:30 AM – 12:00 PM ET  (primary killzone)
# NY PM : 1:30 PM –  4:00 PM ET

ASIA_START   = time(18, 0)
ASIA_END     = time(0,  0)   # midnight — spans two calendar days
LONDON_START = time(2,  0)
LONDON_END   = time(5,  0)
NY_START     = time(9, 30)
NY_END       = time(16, 0)

# Big candle: range > this multiple of ATR(14) on 15m
BIG_CANDLE_ATR_MULT = 2.5

_FF_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.forexfactory.com/",
}


# ── Data fetching ──────────────────────────────────────────────────────────────

def _dl(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    return df.sort_index()


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna()


def fetch_data() -> dict[str, dict[str, pd.DataFrame]]:
    print("  Downloading market data...")
    out = {}
    for asset, ticker in [("NQ", "NQ=F"), ("ES", "ES=F")]:
        df_1m  = _dl(ticker, "1m",  "7d")
        df_5m  = _dl(ticker, "5m",  "60d")
        df_15m = _dl(ticker, "15m", "60d")
        df_1h  = _dl(ticker, "1h",  "730d")
        df_3m  = _resample(df_1m, "3min") if not df_1m.empty else pd.DataFrame()
        df_4h  = _resample(df_1h, "4h")   if not df_1h.empty else pd.DataFrame()
        out[asset] = {
            "1m": df_1m, "3m": df_3m, "5m": df_5m,
            "15m": df_15m, "1h": df_1h, "4h": df_4h,
        }
        bars = {k: len(v) for k, v in out[asset].items()}
        print(f"    {asset}: {bars}")
    return out


# ── Economic calendar ──────────────────────────────────────────────────────────

def fetch_calendar_week() -> list[dict]:
    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=5, headers=_FF_HEADERS,
        )
        raw = r.json()
        events = []
        for e in raw:
            if e.get("country") != "USD":
                continue
            if e.get("impact") not in ("High", "Medium"):
                continue
            events.append({
                "date":     e.get("date", "")[:10],
                "time":     e.get("date", "")[11:16],
                "title":    e.get("title", ""),
                "impact":   e.get("impact", ""),
                "forecast": e.get("forecast", "") or "-",
                "previous": e.get("previous", "") or "-",
            })
        return events
    except Exception:
        return []


# ── Session H/L extraction ─────────────────────────────────────────────────────

def session_hl(df: pd.DataFrame, trading_day: date,
               start_t: time, end_t: time,
               spans_midnight: bool = False) -> tuple[float, float] | None:
    """
    Extract H/L for a session window on (or before) a given trading day.
    spans_midnight=True means start_t is prior calendar day (e.g. Asia 6pm–midnight).
    """
    if spans_midnight:
        # e.g. Asia: 6pm prior calendar day → midnight trading day
        d_start = datetime.combine(trading_day - timedelta(days=1), start_t, tzinfo=ET)
        d_end   = datetime.combine(trading_day, time(0, 0), tzinfo=ET)
    else:
        d_start = datetime.combine(trading_day, start_t, tzinfo=ET)
        d_end   = datetime.combine(trading_day, end_t,   tzinfo=ET)

    window = df[(df.index >= d_start) & (df.index < d_end)]
    if len(window) < 2:
        return None
    return float(window["high"].max()), float(window["low"].min())


# ── Sweep detection ────────────────────────────────────────────────────────────

def detect_sweeps(df_1m: pd.DataFrame, trading_day: date,
                  asia_hl: tuple | None, london_hl: tuple | None) -> list[dict]:
    """
    During the NY session, detect when price takes out Asia or London H/L
    and then closes back inside (FBO = manipulation sweep).
    """
    ny_start = datetime.combine(trading_day, NY_START, tzinfo=ET)
    ny_end   = datetime.combine(trading_day, NY_END,   tzinfo=ET)
    ny       = df_1m[(df_1m.index >= ny_start) & (df_1m.index < ny_end)]

    if ny.empty:
        return []

    sweeps = []
    levels = []
    if asia_hl:
        levels += [
            ("Asia High",  asia_hl[0],  "bearish"),
            ("Asia Low",   asia_hl[1],  "bullish"),
        ]
    if london_hl:
        levels += [
            ("London High", london_hl[0], "bearish"),
            ("London Low",  london_hl[1], "bullish"),
        ]

    for label, level, sweep_dir in levels:
        for i in range(1, len(ny)):
            prev = ny.iloc[i - 1]
            curr = ny.iloc[i]
            ts   = ny.index[i]

            if sweep_dir == "bearish":
                # wick above level, close back below
                if curr["high"] > level and curr["close"] < level:
                    move = round(curr["high"] - level, 2)
                    reversal = round(curr["high"] - curr["close"], 2)
                    sweeps.append({
                        "time":     ts.strftime("%H:%M"),
                        "type":     f"FBO {label}",
                        "level":    round(level, 2),
                        "extreme":  round(curr["high"], 2),
                        "close":    round(curr["close"], 2),
                        "overshoot_pts": move,
                        "reversal_pts":  reversal,
                        "direction": "short",
                    })
                    break  # one sweep per level per day
            else:
                if curr["low"] < level and curr["close"] > level:
                    move = round(level - curr["low"], 2)
                    reversal = round(curr["close"] - curr["low"], 2)
                    sweeps.append({
                        "time":     ts.strftime("%H:%M"),
                        "type":     f"FBO {label}",
                        "level":    round(level, 2),
                        "extreme":  round(curr["low"], 2),
                        "close":    round(curr["close"], 2),
                        "overshoot_pts": move,
                        "reversal_pts":  reversal,
                        "direction": "long",
                    })
                    break

    return sweeps


# ── Big candle detection ───────────────────────────────────────────────────────

def find_big_candles(df_15m: pd.DataFrame, trading_day: date,
                     atr_mult: float = BIG_CANDLE_ATR_MULT) -> list[dict]:
    atr = (df_15m["high"] - df_15m["low"]).rolling(14).mean()
    ny_start = datetime.combine(trading_day, NY_START, tzinfo=ET)
    ny_end   = datetime.combine(trading_day, NY_END,   tzinfo=ET)
    day_df   = df_15m[(df_15m.index >= ny_start) & (df_15m.index < ny_end)]

    big = []
    for ts, row in day_df.iterrows():
        rng      = row["high"] - row["low"]
        atr_val  = atr.loc[ts] if ts in atr.index else np.nan
        if np.isnan(atr_val) or atr_val == 0:
            continue
        if rng >= atr_val * atr_mult:
            big.append({
                "time":      ts.strftime("%H:%M"),
                "range_pts": round(rng, 2),
                "atr_mult":  round(rng / atr_val, 1),
                "direction": "bull" if row["close"] > row["open"] else "bear",
                "close":     round(row["close"], 2),
            })
    return big


# ── Per-day signal summary ─────────────────────────────────────────────────────

def analyse_day(trading_day: date, data: dict, calendar: list[dict]) -> dict:
    df_1m  = data["NQ"]["1m"]
    df_3m  = data["NQ"]["3m"]
    df_5m  = data["NQ"]["5m"]
    df_15m = data["NQ"]["15m"]
    df_1h  = data["NQ"]["1h"]
    df_5m_es = data["ES"]["5m"]

    # Session H/L
    asia_hl   = session_hl(df_15m, trading_day, ASIA_START, ASIA_END, spans_midnight=True)
    london_hl = session_hl(df_15m, trading_day, LONDON_START, LONDON_END)

    # NY range
    ny_start = datetime.combine(trading_day, NY_START, tzinfo=ET)
    ny_end   = datetime.combine(trading_day, NY_END,   tzinfo=ET)
    ny_15m   = df_15m[(df_15m.index >= ny_start) & (df_15m.index < ny_end)]
    ny_open  = float(ny_15m.iloc[0]["open"])  if not ny_15m.empty else None
    ny_high  = float(ny_15m["high"].max())    if not ny_15m.empty else None
    ny_low   = float(ny_15m["low"].min())     if not ny_15m.empty else None
    ny_close = float(ny_15m.iloc[-1]["close"]) if not ny_15m.empty else None

    # Sweeps
    sweeps = detect_sweeps(df_1m, trading_day, asia_hl, london_hl) if not df_1m.empty else []

    # FBOS / RBOS on 15m (last 3 days of context + today)
    cutoff = datetime.combine(trading_day - timedelta(days=3), time(0), tzinfo=ET)
    df_15_ctx = df_15m[df_15m.index >= cutoff]
    if len(df_15_ctx) >= 30:
        df_15_rbos = find_rbos(df_15_ctx)
        today_15 = df_15_rbos[df_15_rbos.index.date == trading_day]
        fbos_bull_count = int(today_15["fbos_bullish"].sum())
        fbos_bear_count = int(today_15["fbos_bearish"].sum())
        rbos_bull_count = int(today_15["rbos_bullish"].sum())
        rbos_bear_count = int(today_15["rbos_bearish"].sum())
    else:
        fbos_bull_count = fbos_bear_count = rbos_bull_count = rbos_bear_count = 0

    # SMT on 1m (resampled to 3m) — today only
    smt_bull = smt_bear = 0
    if not df_3m.empty:
        today_3m = df_3m[df_3m.index.date == trading_day]
        if len(today_3m) >= 10:
            today_3m_smt = find_smt(today_3m)
            smt_bull = int(today_3m_smt["smt_bullish"].sum())
            smt_bear = int(today_3m_smt["smt_bearish"].sum())

    # DIV (NQ vs ES) on 5m — today only
    div_bull = div_bear = 0
    if not df_5m.empty and not df_5m_es.empty:
        today_5_nq = df_5m[df_5m.index.date == trading_day]
        today_5_es = df_5m_es[df_5m_es.index.date == trading_day]
        if len(today_5_nq) >= 10 and len(today_5_es) >= 10:
            nq_div = find_div(today_5_nq, today_5_es)
            div_bull = int(nq_div["div_bullish"].sum())
            div_bear = int(nq_div["div_bearish"].sum())

    # 1H bias
    today_1h = df_1h[df_1h.index.date == trading_day]
    if len(today_1h) >= 2:
        h_open  = float(today_1h.iloc[0]["open"])
        h_close = float(today_1h.iloc[-1]["close"])
        day_bias = "bullish" if h_close > h_open else "bearish"
    else:
        day_bias = "unknown"

    # Big candles
    big_candles = find_big_candles(df_15m, trading_day)

    # News events for this day
    news = [e for e in calendar if e["date"] == trading_day.isoformat()]

    # AMD regime
    amd_notes = []
    if asia_hl and london_hl:
        if london_hl[0] > asia_hl[0]:
            amd_notes.append("London swept Asia High")
        if london_hl[1] < asia_hl[1]:
            amd_notes.append("London swept Asia Low")
    for sw in sweeps:
        amd_notes.append(f"NY: {sw['type']} @ {sw['time']} → {sw['direction']} reversal ({sw['reversal_pts']}pts)")

    return {
        "date":            trading_day.isoformat(),
        "day_bias":        day_bias,
        "asia_high":       round(asia_hl[0], 2)   if asia_hl   else None,
        "asia_low":        round(asia_hl[1], 2)   if asia_hl   else None,
        "london_high":     round(london_hl[0], 2) if london_hl else None,
        "london_low":      round(london_hl[1], 2) if london_hl else None,
        "ny_open":         round(ny_open, 2)       if ny_open   else None,
        "ny_high":         round(ny_high, 2)       if ny_high   else None,
        "ny_low":          round(ny_low, 2)        if ny_low    else None,
        "ny_close":        round(ny_close, 2)      if ny_close  else None,
        "sweeps":          sweeps,
        "fbos_bull":       fbos_bull_count,
        "fbos_bear":       fbos_bear_count,
        "rbos_bull":       rbos_bull_count,
        "rbos_bear":       rbos_bear_count,
        "smt_bull":        smt_bull,
        "smt_bear":        smt_bear,
        "div_bull":        div_bull,
        "div_bear":        div_bear,
        "big_candles":     big_candles,
        "news":            news,
        "amd_notes":       amd_notes,
        "gex_notes":       "",   # populated via TV replay (Phase 2)
    }


# ── HTML report ────────────────────────────────────────────────────────────────

_CSS = """
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#07090f;--surface:#0d111a;--card:#0f1420;--border:#1a2035;
  --accent:#4f8ef7;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;
  --text:#e2e8f0;--muted:#4a5568;--subtle:#1e293b;
}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',Consolas,monospace;font-size:12px;padding:20px 24px 40px}
h1{font-size:16px;font-weight:700;color:var(--accent);letter-spacing:.15em;margin-bottom:4px;text-transform:uppercase}
.subtitle{font-size:10px;color:var(--muted);margin-bottom:24px;letter-spacing:.1em}
.day-card{background:var(--card);border:1px solid var(--border);border-radius:10px;margin-bottom:16px;overflow:hidden}
.day-head{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;background:rgba(255,255,255,.01)}
.day-date{font-size:13px;font-weight:700;color:var(--text)}
.bias-chip{font-size:9px;font-weight:700;letter-spacing:.1em;padding:2px 7px;border-radius:4px;text-transform:uppercase}
.bias-chip.bullish{background:rgba(34,197,94,.12);color:var(--green)}
.bias-chip.bearish{background:rgba(239,68,68,.12);color:var(--red)}
.bias-chip.unknown{background:rgba(74,85,104,.2);color:var(--muted)}
.day-body{display:grid;grid-template-columns:220px 1fr;gap:0}
.sessions{padding:12px 16px;border-right:1px solid var(--border)}
.session-label{font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;margin-top:10px}
.session-label:first-child{margin-top:0}
.hl-row{display:flex;justify-content:space-between;margin-bottom:2px}
.hl-key{color:var(--muted);font-size:11px}
.hl-val{font-size:11px;color:var(--text)}
.right-col{padding:12px 16px}
.section{margin-bottom:10px}
.sec-title{font-size:9px;font-weight:700;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin-bottom:5px}
.tag{display:inline-block;font-size:9px;font-weight:700;padding:2px 6px;border-radius:3px;margin:1px 2px 1px 0;letter-spacing:.04em}
.tag.sweep-long{background:rgba(34,197,94,.12);color:var(--green);border:1px solid rgba(34,197,94,.2)}
.tag.sweep-short{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.2)}
.tag.fbos{background:rgba(245,158,11,.1);color:var(--yellow);border:1px solid rgba(245,158,11,.2)}
.tag.rbos{background:rgba(79,142,247,.1);color:var(--accent);border:1px solid rgba(79,142,247,.2)}
.tag.smt{background:rgba(168,85,247,.1);color:#a855f7;border:1px solid rgba(168,85,247,.2)}
.tag.div{background:rgba(20,184,166,.1);color:#14b8a6;border:1px solid rgba(20,184,166,.2)}
.tag.news-high{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.25)}
.tag.news-med{background:rgba(245,158,11,.1);color:var(--yellow);border:1px solid rgba(245,158,11,.2)}
.tag.big-candle{background:rgba(255,255,255,.05);color:var(--text);border:1px solid var(--border)}
.amd-note{font-size:11px;color:var(--text);padding:3px 0;border-bottom:1px solid var(--subtle)}
.amd-note:last-child{border-bottom:none}
.gex-note{font-size:11px;color:var(--muted);font-style:italic;padding:4px 0}
.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px}
.s-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 14px}
.s-label{font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}
.s-val{font-size:18px;font-weight:700;color:var(--text)}
"""

def _tag(label: str, cls: str) -> str:
    return f'<span class="tag {cls}">{label}</span>'


def build_report(days: list[dict], generated: str) -> str:
    total_sweeps  = sum(len(d["sweeps"]) for d in days)
    total_fbos    = sum(d["fbos_bull"] + d["fbos_bear"] for d in days)
    total_smt     = sum(d["smt_bull"] + d["smt_bear"] for d in days)
    bull_days     = sum(1 for d in days if d["day_bias"] == "bullish")

    summary = f"""
<div class="summary-grid">
  <div class="s-card"><div class="s-label">Days Reviewed</div><div class="s-val">{len(days)}</div></div>
  <div class="s-card"><div class="s-label">Session Sweeps</div><div class="s-val">{total_sweeps}</div></div>
  <div class="s-card"><div class="s-label">FBOS (15m)</div><div class="s-val">{total_fbos}</div></div>
  <div class="s-card"><div class="s-label">SMT Signals</div><div class="s-val">{total_smt}</div></div>
</div>"""

    cards = []
    for d in days:
        bias_cls = d["day_bias"]
        date_str = datetime.fromisoformat(d["date"]).strftime("%A, %d %B %Y")

        # Sessions panel
        def hl_row(k, v):
            return f'<div class="hl-row"><span class="hl-key">{k}</span><span class="hl-val">{v if v else "--"}</span></div>'

        sessions_html = f"""
        <div class="session-label">Asia (6pm–midnight ET)</div>
        {hl_row("High", d["asia_high"])}
        {hl_row("Low",  d["asia_low"])}
        <div class="session-label">London (2am–5am ET)</div>
        {hl_row("High", d["london_high"])}
        {hl_row("Low",  d["london_low"])}
        <div class="session-label">NY Session</div>
        {hl_row("Open",  d["ny_open"])}
        {hl_row("High",  d["ny_high"])}
        {hl_row("Low",   d["ny_low"])}
        {hl_row("Close", d["ny_close"])}
        """

        # AMD notes
        amd_html = ""
        if d["amd_notes"]:
            amd_html = "".join(f'<div class="amd-note">{n}</div>' for n in d["amd_notes"])
        else:
            amd_html = '<div class="amd-note" style="color:var(--muted)">No session sweeps detected</div>'

        # Signal tags
        tags = []
        for sw in d["sweeps"]:
            cls = "sweep-long" if sw["direction"] == "long" else "sweep-short"
            tags.append(_tag(f"{sw['type']} {sw['time']}", cls))
        if d["fbos_bull"]:  tags.append(_tag(f"FBOS ↑ ×{d['fbos_bull']}", "fbos"))
        if d["fbos_bear"]:  tags.append(_tag(f"FBOS ↓ ×{d['fbos_bear']}", "fbos"))
        if d["rbos_bull"]:  tags.append(_tag(f"RBOS ↑ ×{d['rbos_bull']}", "rbos"))
        if d["rbos_bear"]:  tags.append(_tag(f"RBOS ↓ ×{d['rbos_bear']}", "rbos"))
        if d["smt_bull"]:   tags.append(_tag(f"SMT ↑ ×{d['smt_bull']}", "smt"))
        if d["smt_bear"]:   tags.append(_tag(f"SMT ↓ ×{d['smt_bear']}", "smt"))
        if d["div_bull"]:   tags.append(_tag(f"DIV ↑ ×{d['div_bull']}", "div"))
        if d["div_bear"]:   tags.append(_tag(f"DIV ↓ ×{d['div_bear']}", "div"))

        # News
        news_html = ""
        if d["news"]:
            for n in d["news"]:
                cls = "news-high" if n["impact"] == "High" else "news-med"
                news_html += _tag(f"{n['time']} {n['title']}", cls)

        # Big candles
        bc_html = ""
        for bc in d["big_candles"]:
            dir_sym = "↑" if bc["direction"] == "bull" else "↓"
            bc_html += _tag(f"{bc['time']} {dir_sym} {bc['range_pts']}pts ({bc['atr_mult']}×ATR)", "big-candle")

        # GEX
        gex_html = f'<div class="gex-note">{d["gex_notes"] if d["gex_notes"] else "GEX levels not yet populated — run TV replay for this date"}</div>'

        right_html = f"""
        <div class="section">
          <div class="sec-title">AMD / Session Sweeps</div>
          {amd_html}
        </div>
        {"<div class='section'><div class='sec-title'>Signals (15m FBOS/RBOS · 3m SMT · 5m DIV)</div>" + "".join(tags) + "</div>" if tags else ""}
        {"<div class='section'><div class='sec-title'>News Events</div>" + news_html + "</div>" if news_html else ""}
        {"<div class='section'><div class='sec-title'>Big Candles (potential news reaction)</div>" + bc_html + "</div>" if bc_html else ""}
        <div class="section">
          <div class="sec-title">GEX Suite Levels</div>
          {gex_html}
        </div>
        """

        cards.append(f"""
<div class="day-card">
  <div class="day-head">
    <div class="day-date">{date_str}</div>
    <div class="bias-chip {bias_cls}">{bias_cls}</div>
  </div>
  <div class="day-body">
    <div class="sessions">{sessions_html}</div>
    <div class="right-col">{right_html}</div>
  </div>
</div>""")

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><title>AMD Session Review — NQ</title>
<style>{_CSS}</style>
</head><body>
<h1>AMD Session Review — NQ Futures</h1>
<div class="subtitle">Generated {generated} · Sessions: Asia 6pm–midnight ET | London 2–5am ET | NY 9:30am–4pm ET</div>
{summary}
{"".join(cards)}
</body></html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n=== AMD Session Review — NQ ===\n")

    data     = fetch_data()
    calendar = fetch_calendar_week()
    print(f"  Calendar events loaded: {len(calendar)}")

    # Trading days: last 10 sessions (2 weeks)
    df_ref = data["NQ"]["15m"]
    if df_ref.empty:
        print("ERROR: no 15m data available.")
        return

    all_dates = sorted(set(df_ref.index.date))
    trading_days = [d for d in all_dates if d.weekday() < 5][-10:]
    print(f"  Trading days: {trading_days[0]} to {trading_days[-1]}\n")

    days = []
    for td in trading_days:
        print(f"  Analysing {td}...")
        result = analyse_day(td, data, calendar)
        days.append(result)
        sweep_str = f"{len(result['sweeps'])} sweep(s)" if result["sweeps"] else "no sweeps"
        print(f"    bias={result['day_bias']}  {sweep_str}  "
              f"FBOS={result['fbos_bull']+result['fbos_bear']}  "
              f"SMT={result['smt_bull']+result['smt_bear']}")

    # Save HTML
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = build_report(days, generated)
    out_html = RESULTS_DIR / "amd_review.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\n  Saved -> {out_html}")

    # Save JSON for Phase 2 GEX overlay
    out_json = RESULTS_DIR / "amd_review.json"
    out_json.write_text(json.dumps(days, indent=2, default=str))
    print(f"  Saved -> {out_json}")

    import webbrowser
    webbrowser.open(out_html.as_uri())
    print("  Opened in browser.\n")


if __name__ == "__main__":
    main()
